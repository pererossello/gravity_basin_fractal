import copy

import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit

import plot_utils as pu

class Body:
    def __init__(self, mass, position, velocity, dtype=np.float64):

        if isinstance(position, (list, tuple)) or isinstance(velocity, (list, tuple)):
            position = np.array(position)
            velocity = np.array(velocity)

        if not (position.shape == (2,) and velocity.shape == (2,)):
            raise ValueError("Position and velocity must be two-dimensional.")

        self.mass = dtype(mass)
        self.position = position.astype(dtype)
        self.velocity = velocity.astype(dtype)


class NBodySimulation:
    def __init__(self, bodies, e=0):

        self.sim_run = False  # True if simulation has been run
        body_list = copy.deepcopy(bodies)
    
        def is_body(bod): return str(type(bod)) == "<class 'utils.Body'>"
        del bodies
        if not all(is_body(body) for body in body_list):
            raise TypeError(
                "All elements in 'bodies' must be instances of the 'Body' class"
            )
        
        self.bodies = body_list
        self.num_bodies = len(body_list)
        self.masses = np.array([body.mass for body in body_list])

        self.e = e
        self.interaction = lambda r: -r * (r**2 + self.e**2)**(-3/2)

        if self.e != 0:
            self.max_acc = 2 / (3*np.sqrt(3) * self.e**2)
        else:
            self.max_acc = np.inf

        self.nonzero_mass_indices = list(np.nonzero(self.masses)[0])
        nonzero_mass_indices_set = set(self.nonzero_mass_indices)
        all_indices = set(range(self.num_bodies))
        self.massless_indices = list(all_indices - nonzero_mass_indices_set)

    def net_fields(self):
        
        fields = np.zeros((self.num_bodies, 2))

        i_already = []
        for i in self.nonzero_mass_indices:
            for j in self.nonzero_mass_indices:  # prevents double counting
                if j in i_already or j == i:
                    continue
                body_i = self.bodies[i]
                body_j = self.bodies[j]
                displacement = body_i.position - body_j.position
                r = np.linalg.norm(displacement)
                force_magnitude = body_i.mass * body_j.mass * self.interaction(r) 
                force_vector = force_magnitude * displacement / r
                fields[i] += force_vector / body_i.mass
                fields[j] -= force_vector / body_j.mass

            i_already.append(i)

        # Compute the effect of massive bodies on massless bodies
        for i in self.nonzero_mass_indices:
            for j in self.massless_indices:
                body_i = self.bodies[i]
                body_j = self.bodies[j]

                displacement = body_i.position - body_j.position
                r = np.linalg.norm(displacement)
                acc = body_i.mass * self.interaction(r) * displacement / r
                fields[j] -= acc

        return fields
    
    #@numba.njit
    def leapfrog_step(self, pos, vel, acc, dt):
        # Perform a half-step velocity update
        vel_half = vel + 0.5 * acc * dt
        # Update position
        pos_next = pos + vel_half * dt
        return pos_next, vel_half

    def update_positions_and_velocities(self, dt):

        # Update fields (accelerations) based on new positions
        if self.step == 0:
            self.fields = self.net_fields()

        for i, body in enumerate(self.bodies):
            body.position, body.velocity = self.leapfrog_step(body.position, body.velocity, self.fields[i], dt)

        self.fields = self.net_fields()

        for i, body in enumerate(self.bodies):
            _ , body.velocity = self.leapfrog_step(body.position, body.velocity, self.fields[i], dt)


    def run_simulation(self, duration, time_step):

        num_steps = int(duration / time_step)
        num_bodies = len(self.bodies)
        pos_arr = np.zeros((num_steps, 2, num_bodies))
        # vel_arr = np.zeros((num_steps, 3, num_bodies))

        for step in range(num_steps):
            self.step = step
            for i, body in enumerate(self.bodies):
                pos_arr[step, :, i] = body.position
                # vel_arr[step, :, i] = body.velocity

            perc = step/num_steps * 100
            print(f'\r{perc:.2f}%', end='')

            self.update_positions_and_velocities(time_step)

        self.positions = pos_arr
        # self.velocities = vel_arr
        self.sim_run = True
        self.num_steps = num_steps

    def simple_plot(self, step=1):

        fig_size, ratio = 540, 3
        subplots = (1,1)
        Fig = pu.Figure(fig_size=fig_size, ratio=ratio, 
                        subplots=subplots)

        axes = Fig.axes
        axes_flat = Fig.axes_flat
        axes[0][0].plot(self.positions[:, 0], self.positions[:, 1], alpha=1)

        for ax in axes_flat:
            ax.set_aspect('equal')


class two_body:

    def __init__(self, 
                 r_CM=[0, 0, 0], 
                 v_CM=[0, 0, 0],
                 l=[0, 0, 1],
                 ecc=0, 
                 r_0=1, 
                 omega=0, 
                 phi_0=0, 
                 m0=1, m1=1, 
                 G=1):
        
        self.r_CM = np.array(r_CM)
        self.v_CM = np.array(v_CM)
        self.l = np.array(l)
        self.l = self.l / np.linalg.norm(self.l)
        self.e = ecc
        self.r_0 = r_0
        self.omega = omega
        self.phi_0 = phi_0
        self.m0 = m0
        self.m1 = m1
        self.G = G

        self.mu = (m0**-1 + m1**-1) ** -1
        self.M = m0 + m1
        self.L = self.mu * np.sqrt(r_0 * G * self.M)

    def r(self, phi):
        return self.r_0 / (1 + self.e * np.cos(phi))

    def r_dot(self, phi):
        return self.G * self.M * self.mu / self.L * self.e * np.sin(phi)

    def phi_dot(self, phi):
        return self.L / (self.mu * self.r(phi) ** 2)

    def r_vec(self, phi):
        x = self.r(phi) * np.cos(phi)
        y = self.r(phi) * np.sin(phi)

        return np.array([x, y, 0])

    def v_vec(self, phi):
        vx = self.r_dot(phi) * np.cos(phi) - self.r(phi) * np.sin(phi) * self.phi_dot(phi)
        vy = self.r_dot(phi) * np.sin(phi) + self.r(phi) * np.cos(phi) * self.phi_dot(phi)

        return np.array([vx, vy, 0])

    def periapsis_rotation(self, x, y, omega):
        x_rot = x * np.cos(omega) - y * np.sin(omega)
        y_rot = x * np.sin(omega) + y * np.cos(omega)
        return x_rot, y_rot

    def orbit_rotation(self, vec, n, theta):
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)

        vec_rot = np.matmul(R, vec)

        return vec_rot
    

    def get_bodies(self):
        r_vec_0 = self.r_vec(self.phi_0)
        v_vec_0 = self.v_vec(self.phi_0)

        r_vec_0[:2] = self.periapsis_rotation(*r_vec_0[:2], self.omega)
        v_vec_0[:2] = self.periapsis_rotation(*v_vec_0[:2], self.omega)

        if not np.array_equal(self.l, np.array([0, 0, 1])):
            if np.array_equal(self.l, np.array([0, 0, -1])):
                n = np.array([1, 0, 0])  # Set n to x-axis (or any other non-zero vector) when l is anti-parallel to the z-axis
            else:
                n = np.array([-self.l[1], self.l[0], 0])
                n = n / np.linalg.norm(n)
            theta = np.arccos(self.l[2])
            r_vec_0 = self.orbit_rotation(r_vec_0, n, theta)
            v_vec_0 = self.orbit_rotation(v_vec_0, n, theta)

        r0_vec_0 = -self.m1 / self.M * r_vec_0
        r1_vec_0 = self.m0 / self.M * r_vec_0

        v0_vec_0 = -self.m1 / self.M * v_vec_0
        v1_vec_0 = self.m0 / self.M * v_vec_0

        r_vec_0 = r_vec_0 + self.r_CM
        v_vec_0 = v_vec_0 + self.v_CM

        r0_vec_0 = r0_vec_0 + self.r_CM
        r1_vec_0 = r1_vec_0 + self.r_CM

        v0_vec_0 = v0_vec_0 + self.v_CM
        v1_vec_0 = v1_vec_0 + self.v_CM

        body1 = Body(self.m0, r0_vec_0[:2], v0_vec_0[:2])
        body2 = Body(self.m1, r1_vec_0[:2], v1_vec_0[:2])

        return body1, body2
    

    def get_period(self):
        if self.e < 1:
            a = self.r_0 / (1 - self.e**2)
            M = self.m0 + self.m1
            T = 2 * np.pi * np.sqrt(a**3 / (self.G * M))

        else:
            T = np.inf

        return T



@jit(nopython=True)
def get_acc(positions, masses, loc, hit_indices, step):
    pos = positions[step, :, :]  # shape (2, num_bodies)

    # Flatten loc and hit_indices for compatible indexing
    loc_flat = loc.reshape(-1, loc.shape[-1])
    hit_indices_flat = hit_indices.ravel()

    # Only include points that have not been hit
    not_hit_mask_flat = hit_indices_flat == -1
    loc_filtered = loc_flat[not_hit_mask_flat]

    # Broadcasting to compute the differences
    r = loc_filtered[:, None, :] - pos.T[None, :, :]

    # Manually compute the norms
    norms = np.sqrt(r[:, :, 0]**2 + r[:, :, 1]**2)
    norms_safe = np.where(norms == 0, np.inf, norms)

    # Compute acceleration components
    a = -masses / norms_safe**3
    a = (a[:, :, None] * r).sum(axis=1)

    # Initialize full acceleration array with zeros
    full_a_flat = np.zeros_like(loc_flat)
    full_a_flat[not_hit_mask_flat] = a

    # Reshape back to original loc shape
    full_a = full_a_flat.reshape(loc.shape)

    return full_a

@numba.njit
def leapfrog_step(pos, vel, acc, dt):
    # Perform a half-step velocity update
    vel_half = vel + 0.5 * acc * dt
    # Update position
    pos_next = pos + vel_half * dt
    return pos_next, vel_half


class BasinFractal():

    def __init__(self, masses, radii, positions):

        self.masses = np.array(masses)
        self.radii = np.array(radii)
        self.positions = positions
        self.num_bodies = len(masses)
        self.N = positions.shape[0]
 
    def set_grid(self, n, m, dx, dy, x0=0, y0=0, radi=0.02):

        self.n = n
        self.m = m
        self.radi = radi

        x_min, x_max = -dx, dx
        y_min, y_max = -dy, dy
    
        x = np.linspace(x_min+x0, x_max+x0, n)
        y = np.linspace(y_min+y0, y_max+y0, m)
        X, Y = np.meshgrid(x, y, indexing='ij')

        self.loc = np.stack((X, Y), axis=-1)

        self.hit_indices = np.full((self.n, self.m), -1)
        self.hit_time = np.full((self.n, self.m), -1)






    def check_hit(self, frame):
            pos = self.positions[frame, :, :]  # shape (2, num_bodies)

            # Expand loc and positions to 4D for vectorized subtraction
            # loc_expanded shape: (n, m, 1, 2), pos_expanded shape: (1, 1, num_bodies, 2)
            loc_expanded = self.loc[:, :, np.newaxis, :]
            pos_expanded = pos.T[np.newaxis, np.newaxis, :, :]

            # Vectorized subtraction to get r, shape: (n, m, num_bodies, 2)
            r = loc_expanded - pos_expanded

            # Calculate distances, shape: (n, m, num_bodies)
            distances = np.linalg.norm(r, axis=3)

            # Compare distances with radii (broadcasted), shape: (n, m, num_bodies)
            hit_mask = distances < (self.radii + self.radi)

            # Update hit_arr and hit_indices
            # We use logical_or to update the hit_arr, ensuring we keep previous hits
            # For hit_indices, we need to choose the first hit in case of multiple hits
            for b in range(self.num_bodies):
                body_hit_mask = hit_mask[:, :, b]
                self.hit_arr |= body_hit_mask

                # Update hit_indices for the current body, avoid overwriting previous hits
                not_hit_yet_mask = (self.hit_indices == -1)  # Assuming -1 indicates no hit yet
                self.hit_indices = np.where(body_hit_mask & not_hit_yet_mask, b, self.hit_indices)
                self.hit_time = np.where(body_hit_mask & not_hit_yet_mask, self.k, self.hit_time)



    def run(self, dt, frame_start=0, max_iters=1000, make_pos_arr=False):

        max_iters

        self.hit_arr = np.full((self.n, self.m), False, dtype=bool)

        self.dt = dt
        self.vel = np.zeros_like(self.loc)

        all_hit = np.all(self.hit_arr) 

        if make_pos_arr:
            self.pos_arr = []

        l = frame_start
        self.k = 0
        while (not all_hit) and (self.k<max_iters):

            perc = self.k/max_iters * 100
            print(f'\r{perc:.2f}%', end='')

            l = l % self.N

            self.check_hit(l)

            if l == frame_start:
                self.a = get_acc(self.positions, self.masses, self.loc, self.hit_indices, l)

            # Apply updates only to not-hit points
            not_hit_mask = self.hit_indices == -1
            self.loc[not_hit_mask], self.vel[not_hit_mask] = leapfrog_step(self.loc[not_hit_mask], self.vel[not_hit_mask], self.a[not_hit_mask], self.dt)

            self.a = get_acc(self.positions, self.masses, self.loc, self.hit_indices, l)
            _ , self.vel[not_hit_mask] = leapfrog_step(self.loc[not_hit_mask], self.vel[not_hit_mask], self.a[not_hit_mask], self.dt)

            if make_pos_arr:
                self.pos_arr.append(self.loc)

            self.loc[self.hit_arr] = np.nan



            l += 1
            self.k += 1

        if make_pos_arr:
            self.pos_arr = np.array(self.pos_arr)



class BasinFractalOld():

    def __init__(self, masses, radii, positions):

        self.masses = np.array(masses)
        self.radii = np.array(radii)
        self.positions = positions
        self.num_bodies = len(masses)
        self.N = positions.shape[0]
 
    def set_grid(self, n, m, dx, dy, x0=0, y0=0, radi=0.02):

        self.n = n
        self.m = m
        self.radi = radi

        x_min, x_max = -dx, dx
        y_min, y_max = -dy, dy
    
        x = np.linspace(x_min+x0, x_max+x0, n)
        y = np.linspace(y_min+y0, y_max+y0, m)
        X, Y = np.meshgrid(x, y, indexing='ij')

        self.loc = np.stack((X, Y), axis=-1)

        self.hit_indices = np.full((self.n, self.m), -1)
        self.hit_time = np.full((self.n, self.m), -1)

    def get_acc(self, step):
        pos = self.positions[step, :, :]  # shape (2, num_bodies)

        # Broadcasting to compute the differences: shape becomes (n, m, num_bodies, 2)
        r = self.loc[:, :, None, :] - pos.T[None, None, :, :]

        # Calculate norms along the last dimension and avoid division by zero
        norms = np.linalg.norm(r, axis=3)
        norms_safe = np.where(norms == 0, np.inf, norms)

        # Compute acceleration components
        a = -self.masses / norms_safe**3
        a = (a[:, :, :, None] * r).sum(axis=2)

        return a


    def leapfrog_step(self, pos, vel, acc, dt):
        # Perform a half-step velocity update
        vel_half = vel + 0.5 * acc * dt
        # Update position
        pos_next = pos + vel_half * dt
        return pos_next, vel_half


    def check_hit(self, frame):
            pos = self.positions[frame, :, :]  # shape (2, num_bodies)

            # Expand loc and positions to 4D for vectorized subtraction
            # loc_expanded shape: (n, m, 1, 2), pos_expanded shape: (1, 1, num_bodies, 2)
            loc_expanded = self.loc[:, :, np.newaxis, :]
            pos_expanded = pos.T[np.newaxis, np.newaxis, :, :]

            # Vectorized subtraction to get r, shape: (n, m, num_bodies, 2)
            r = loc_expanded - pos_expanded

            # Calculate distances, shape: (n, m, num_bodies)
            distances = np.linalg.norm(r, axis=3)

            # Compare distances with radii (broadcasted), shape: (n, m, num_bodies)
            hit_mask = distances < (self.radii + self.radi)

            # Update hit_arr and hit_indices
            # We use logical_or to update the hit_arr, ensuring we keep previous hits
            # For hit_indices, we need to choose the first hit in case of multiple hits
            for b in range(self.num_bodies):
                body_hit_mask = hit_mask[:, :, b]
                self.hit_arr |= body_hit_mask

                # Update hit_indices for the current body, avoid overwriting previous hits
                not_hit_yet_mask = (self.hit_indices == -1)  # Assuming -1 indicates no hit yet
                self.hit_indices = np.where(body_hit_mask & not_hit_yet_mask, b, self.hit_indices)
                self.hit_time = np.where(body_hit_mask & not_hit_yet_mask, self.k, self.hit_time)



    def run(self, dt, frame_start=0, max_iters=1000, make_pos_arr=False):

        max_iters

        self.hit_arr = np.full((self.n, self.m), False, dtype=bool)

        self.dt = dt
        self.vel = np.zeros_like(self.loc)

        all_hit = np.all(self.hit_arr) 


        if make_pos_arr:
            self.pos_arr = []


        l = frame_start
        self.k = 0
        while (not all_hit) and (self.k<max_iters):

            perc = self.k/max_iters * 100
            print(f'\r{perc:.2f}%', end='')

            l = l % self.N

            self.check_hit(l)

            if l == frame_start:
                self.a = self.get_acc(l)

            self.loc, self.vel = self.leapfrog_step(self.loc, self.vel, self.a, self.dt)
            self.a = self.get_acc(l)
            _ , self.vel = self.leapfrog_step(self.loc, self.vel, self.a, self.dt)

            l += 1
            self.k += 1

            self.loc[self.hit_arr] = np.nan

            if make_pos_arr:
                self.pos_arr.append(self.loc)
        if make_pos_arr:
            self.pos_arr = np.array(self.pos_arr)