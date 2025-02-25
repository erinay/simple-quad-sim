import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter
import matplotlib.pyplot as plt
import pandas as pd
import mlmodel
import torch

def quat_mult(q, p):
    # q * p
    # p,q = [w x y z]
    return np.array(
        [
            p[0] * q[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] + q[2] * p[3] - q[3] * p[2],
            q[2] * p[0] + q[0] * p[2] + q[3] * p[1] - q[1] * p[3],
            q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1],
        ]
    )
    
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_vectors(v_from, v_to):
    v_from = normalized(v_from)
    v_to = normalized(v_to)
    v_mid = normalized(v_from + v_to)
    q = np.array([np.dot(v_from, v_mid), *np.cross(v_from, v_mid)])
    return q

def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm

NO_STATES = 13
IDX_POS_X = 0
IDX_POS_Y = 1
IDX_POS_Z = 2
IDX_VEL_X = 3
IDX_VEL_Y = 4
IDX_VEL_Z = 5
IDX_QUAT_W = 6
IDX_QUAT_X = 7
IDX_QUAT_Y = 8
IDX_QUAT_Z = 9
IDX_OMEGA_X = 10
IDX_OMEGA_Y = 11
IDX_OMEGA_Z = 12
IDX_TIME = 13

LAMBDA = 0.1 #gain for regularization ahat
R = np.linalg.inv(10 *np.eye(3))
Q = 0.1 *np.eye(9)
LAM = np.eye(3)*1 #p.d. gain for s
# LAM[0,0] = 1
# LAM[1,1] = 1
LAM[2,2] = 0.01
K = np.eye(3)*5
# K[0,0] = 1
# K[1,1] = 1
# K[2,2] = 1

# Load Model
dim_a = 3
features = ['v', 'q', 'pwm']
label = 'fa'
dataset = 'neural-fly' 
dataset_folder = 'data/training'
hover_pwm_ratio = 1.
modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}" # 'intel-aero_fa-num-Tsp_v-q-pwm'
stopping_epoch = 300
final_model = mlmodel.load_model(modelname = modelname + '-epoch-' + str(stopping_epoch), modelfolder='./neural-fly/models/')


class Robot:
    
    '''
    frames:
        B - body frame
        I - inertial frame
    states:
        p_I - position of the robot in the inertial frame (state[0], state[1], state[2])
        v_I - velocity of the robot in the inertial frame (state[3], state[4], state[5])
        q - orientation of the robot (w=state[6], x=state[7], y=state[8], z=state[9])
        omega - angular velocity of the robot (state[10], state[11], state[12])
    inputs:
        omega_1, omega_2, omega_3, omega_4 - angular velocities of the motors
    '''
    def __init__(self):
        self.m = 1.0 # mass of the robot
        self.arm_length = 0.25 # length of the quadcopter arm (motor to center)
        self.height = 0.05 # height of the quadcopter
        self.body_frame = np.array([(self.arm_length, 0, 0, 1),
                                    (0, self.arm_length, 0, 1),
                                    (-self.arm_length, 0, 0, 1),
                                    (0, -self.arm_length, 0, 1),
                                    (0, 0, 0, 1),
                                    (0, 0, self.height, 1)])

        self.J = 0.025 * np.eye(3) # [kg m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.constant_thrust = 10e-4
        self.constant_drag = 10e-6
        self.omega_motors = np.array([0.0, 0.0, 0.0, 0.0])
        self.state = self.reset_state_and_input(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.time = 0.0
        self.data_log = []
        # Set a to adapt
        self.a_hat= np.array([0.1607617, 0.21795802,  0.18300827,
            0.29939878,0.25495464,0.22978155,
            4.50169764, 4.99805797,-0.06836199])
        # self.a_hat = np.array([0.1607617, 0.29939878, 4.50169764, 0.21795802, 0.25495464, 4.99805797, 0.18300827, 0.22978155, 0.06836199])
        self.fa = np.zeros([0,0,0])
        self.phi = None
        self.s = np.array([[0],[0],[0]])
        self.P = 0.1 *np.eye(9)
        self.P[2,2] =0.001

    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        return state0

    def update(self, omegas_motor, dt, wind_force):
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        thrust = self.constant_thrust * np.sum(omegas_motor**2)
        f_b = np.array([0, 0, thrust])
        
        tau_x = self.constant_thrust * (omegas_motor[3]**2 - omegas_motor[1]**2) * 2 * self.arm_length
        tau_y = self.constant_thrust * (omegas_motor[2]**2 - omegas_motor[0]**2) * 2 * self.arm_length
        tau_z = self.constant_drag * (omegas_motor[0]**2 - omegas_motor[1]**2 + omegas_motor[2]**2 - omegas_motor[3]**2)
        tau_b = np.array([tau_x, tau_y, tau_z])

        v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81]) + 1/self.m * wind_force
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I
        
        # Update ahat
        phi = self.phi.detach().numpy()
        phiarr  = np.zeros((3,9))
        phiarr[0,0:3] = phi
        phiarr[1,3:6] = phi
        phiarr[2,6:9] = phi
        adot = -LAMBDA * self.a_hat - self.P @ phiarr.T @ R @(phiarr@self.a_hat - wind_force) + np.reshape(self.P @ phiarr.T @ self.s, (9,))
        pdot = -2*LAMBDA*self.P + Q -self.P @ phiarr.T @ R @ phiarr @ self.P
        self.P += pdot *dt
        self.a_hat += adot*dt
        print(phiarr@self.a_hat - wind_force)

        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.
        self.time += dt

    def control(self, p_d_I, pd, v_d_I, a_d_I):
        #_I means inertial frame
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]


        xdata = np.concatenate((v_I, q, self.omega_motors))
        xdata = torch.from_numpy(xdata)
        phi_net = final_model.phi
        self.phi = phi_net(xdata)

        # Position controller. (Add neural-fly here)
        if pd:
            k_p = 5.0
            k_d = 7.0
            v_r = - k_p * (p_I - p_d_I)
            a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81])
            f = self.m * a  
        else:
            phi = self.phi.detach().numpy()
            phi=0
            phiarr  = np.zeros((3,9))
            phiarr[0,0:3] = phi
            phiarr[1,3:6] = phi
            phiarr[2,6:9] = phi
            ff_term = phiarr @ self.a_hat

            ar = a_d_I - LAM @ (v_d_I-v_I)
            vr = v_d_I - LAM @ (p_d_I-p_I)
            self.s = v_I - vr 
            # f = self.m * ar + np.array([0,0,9.8]) - K @ self.s -ff_term

            k_p = 5.0
            k_d = 7.0
            v_r = - k_p * (p_I - p_d_)
            a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81])

            f = self.m * a + np.array([0,0,9.8]) - ff_term

        f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f
        thrust = np.max([0, f_b[2]])
        
        # Attitude controller. (derived in pset 1)
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b)
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))
        self.omega_motor = omega_motor
        

        # des_state_cur = np.zeros((1,6))
        # des_state_cur[0][0:3] = p_d_I
        # des_state_cur[0][3:6] = v_r
        # self.des_states = np.append(self.des_states, [des_state_cur], axis=0)

        return omega_motor
    
    def save_data(self, state_des, pwm,t,fa):
        append_state = np.zeros(NO_STATES+1).T
        p = self.state[IDX_POS_X:IDX_POS_Z+1].tolist()
        v = self.state[IDX_VEL_X:IDX_VEL_Z+1].tolist()
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1].tolist()
        # qrot = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        w = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1].tolist()
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().tolist()
        p_d = state_des.tolist()
        k_p = 1.0
        v_d = - k_p * (self.state[IDX_POS_X:IDX_POS_Z+1]- state_des)
        v_d = v_d.tolist()
        pwm = pwm.tolist()
        fa = fa.tolist()

        self.data_log.append([t,p,p_d,v,v_d,q,R,w,pwm,fa])

PLAYBACK_SPEED = 1
CONTROL_FREQUENCY = 200 # The Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

def get_pos_full_quadcopter(quad):
    """ position returns a 3 x 6 matrix 
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
    """
    origin = quad.state[IDX_POS_X:IDX_POS_Z+1]
    quat = quad.state[IDX_QUAT_W:IDX_QUAT_Z+1]
    rot = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()
    wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
    quadBodyFrame = quad.body_frame.T
    quadWorldFrame = wHb.dot(quadBodyFrame)
    pos_full_quad = quadWorldFrame[0:3]
    return pos_full_quad

def control_propellers(quad, pd):
    t = quad.time
    T = 5#np.pi/1.5
    r = 2*np.pi * t / T
    p_d_I = np.array([np.cos(r), 0.5*np.sin(2*r), 0.5*t])
    v_d_I = np.array([-2*np.pi/T*np.sin(r), 2*np.pi/T*np.cos(2*r), 0.5])
    a_d_I = np.array([-4*np.pi**2/T**2*np.cos(r), -8*np.pi**2/T**2*np.sin(2*r), 0])
    
    # t = quad.time
    # T = 7
    # r = 2*np.pi * t / T
    # p_d_I = np.array([np.cos(r/2), np.sin(r), 0.5*t])
    # v_d_I = np.array([-np.pi/T*np.sin(r/2), 2*np.pi/T*np.cos(r), 0.5])
    # a_d_I = np.array([-np.pi*np.pi/(T*T)*np.cos(r/2), -4*np.pi*np.pi/(T*T)*np.sin(r), 0.0])
    prop_thrusts = quad.control(p_d_I, pd, v_d_I, a_d_I)
    # prop_thrusts = quad.control(p_d_I = np.array([0,0,1]))
    # Note: for Hover mode, just replace the desired trajectory with [1, 0, 1]
    #calculate wind
    # w_wind = [np.pi, 5, np.pi*2]
    varying=False
    # Uniform Wind
    wind_const = np.array([8,8,0])
    wind_max = np.array([4,4,4])
    if t<0:
        F_wind = np.array([0,0,0])
    else:
        if not varying:
            F_wind = wind_max
        else:
            #Sinusoidal Wind
            F_wind = wind_const+wind_max*np.sin(t)
            # F_wind = [wind_max[0]*np.sin(w_wind[0]*t+phi),wind_max[1]*np.sin(w_wind[1]*t+phi),wind_max[2]*np.cos(w_wind[2]*t+phi)]

    F_wind = F_wind+np.random.rand(1,3)[0]*2-1
    quad.update(prop_thrusts, dt, F_wind)
    quad.save_data(p_d_I, prop_thrusts,t, F_wind)


def main():
    quadcopter = Robot()
    quadcopter2 = Robot()
    def control_loop(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter, False)
            if quadcopter.time>15:
                print('stop')
        return get_pos_full_quadcopter(quadcopter)
    def control_loop2(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter2, True)
            if quadcopter.time>15:
                print('stop')
        return get_pos_full_quadcopter(quadcopter2)
    
    plotter = QuadPlotter()
    plotter.plot_animation(control_loop)

    # Make pandas data frame and export to data folders
    df = pd.DataFrame(quadcopter.data_log, columns=['t','p','p_d', 'v', 'v_d','q','R', 'w', 'pwm', 'fa'])
    # df.to_csv('neural-fly/data/experiment/custom_f8_8p2sint_baseline_82wind.csv')
    plotter2 = QuadPlotter()
    plotter2.plot_animation(control_loop2)
    df2 = pd.DataFrame(quadcopter2.data_log, columns=['t','p','p_d', 'v', 'v_d','q','R', 'w', 'pwm', 'fa'])

    df[['x', 'y', 'z']] = pd.DataFrame(df['p'].tolist(), index=df.index)
    df[['xp', 'yp', 'zp']] = pd.DataFrame(df['p_d'].tolist(), index=df.index)

    df2[['x', 'y', 'z']] = pd.DataFrame(df2['p'].tolist(), index=df2.index)
    df2[['xp', 'yp', 'zp']] = pd.DataFrame(df2['p_d'].tolist(), index=df2.index)


    #PLOT X VS Y FOR POSTIION CONTROLLER VS FF WIND TERM
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['x'],df['y'],df['z'])
    ax.plot(df['xp'],df['yp'],df['zp'])
    ax.plot(df2['x'],df2['y'],df2['z'])
    ax.plot(df2['xp'],df2['yp'],df2['zp'])
    ax.legend(["Wind term","Wind dersired position", "position control","Wind dersired position"])

    # fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # # Plot x, y position over time
    # axs[0].plot(quadcopter.all_states[:,NO_STATES], quadcopter.all_states[:,IDX_POS_X], label="x position")
    # axs[0].plot(quadcopter.all_states[:,NO_STATES], quadcopter.all_states[:,IDX_POS_Y], label="y position")
    # axs[0].plot(quadcopter.all_states[:,NO_STATES], quadcopter.all_states[:,IDX_POS_Z], label="z position")
    # axs[0].set_ylabel("Position (m)")
    # axs[0].set_title("Position Over Time")
    # axs[0].legend()

    # # Plot xdot, ydot velocity over time
    # axs[1].plot(quadcopter.all_states[:,NO_STATES], quadcopter.all_states[:,IDX_VEL_X], label="x velocity")
    # axs[1].plot(quadcopter.all_states[:,NO_STATES], quadcopter.all_states[:,IDX_VEL_Y], label="y velocity")
    # axs[1].plot(quadcopter.all_states[:,NO_STATES], quadcopter.all_states[:,IDX_VEL_Z], label="z velocity")    
    # axs[1].set_ylabel("Velocity (m/s)")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_title("Velocity Over Time")
    # axs[1].legend()

    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
