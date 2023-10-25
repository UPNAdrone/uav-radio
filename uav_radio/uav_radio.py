import matplotlib.pyplot as plt
import math
import cmath
import numpy as np

class PathLossCalculator:
    def __init__(self, reference_distance, n=2.0, c=299792458.0):
        self.n = n # 
        self.c = c # Speed of light in meters per second
        self.d0 = reference_distance # meter
        
    # Methods for Path Loss
    def free_space_pl(self, distance, frequency):
        """
        Calculate Free Space Path Loss (FSPL).

        Args:
            distance (float): Distance between transmitter and receiver (meters).
            frequency (float): Signal frequency (Hz)
            
        Returns:
            float: Path Loss value in dB.
            float: Standard deviation value (if applicable).
        """
        
        wavelength = self.c / frequency
        fspl = 20 * math.log10((4 * math.pi * distance) / wavelength)
        return fspl, None



    def log_distance_pl(self, distance, frequency, scenario="open_field", n_custom = None, sigma_custom = None, speed=None, tx_height=None, foliage=0):
        """
        Calculate Log-Distance Path Loss Model.

        Args:
            distance (float): Distance between transmitter and receiver (meters).
            frequency (float): Signal frequency (Hz).
            scenario (string): "open_field", "urban_rural", "lightly_hilly_rural", "urban", "suburban", "hilly_with_ridge", "dry_hilly", "very_mountainous", "fresh_water", "sea", "custom".
            n_custom (float): custom path loss exponent, use with scenario="custom".
            sigma_custom (float): custom standard-deviation, use with scenario="custom".
            speed (float): UAV speed (mph) for "urban" and "suburban" cases.
            tx_height (float): < 1.0 or >= 1.0 (meters depending on whether the TX is standing on the ground or elevated. For "urban" and "suburban" cases.
            foliage (int): either 0 (False) or 1 (True) for "open_field" and "suburban" cases.

        Returns:
            float: Path Loss value in dB.
            float: Standard deviation value (if applicable).
        """
        wavelength = self.c / frequency
        n = 2.0
        sigma = 0.
        pl_0  = 10.0 * math.log10(((4 * math.pi * self.d0)/wavelength)**2)
        
        if scenario == "open_field":
            n = 2.01
            sigma = 0.
            
            if (frequency >= 3.1E+09) and (frequency <= 5.3E+09):
                if speed == 0:
                    if foliage == 0:
                        if tx_height < 1.0:
                            n = 2.9442
                            sigma = 2.799
                        elif tx_height >= 1.0:
                            n = 2.5418
                            sigma = 3.06
                        else:
                            print("Please specify a tx_height to use this model")
                            return 0,0
                    elif foliage ==1:
                        n = 2.6471
                        sigma = 3.37
                if speed == 20: # speed is in mph
                    if foliage == 0:
                        if tx_height < 1.0:
                            n = 2.9423
                            sigma = 3.44
                        elif tx_height >= 1.0:
                            n = 2.6621
                            sigma = 3.91
                        else:
                            print("Please specify a tx_height to use this model")
                            return 0,0
                    elif foliage ==1:
                        n = 2.6533
                        sigma = 4.02
                            
                        
        elif scenario == "urban_rural":
            n = 4.1
            sigma = 5.24
        elif scenario == "lightly_hilly_rural":
            print("Not implemented yet")
            return 0,0
        elif scenario == "urban":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.7
                sigma = 2.6
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09):
                n = 2.0
                sigma = 3.2
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09] or [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "suburban":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.7
                sigma = 3.1
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09) and (speed == None):
                n = 1.5
                sigma = 2.9
            elif (frequency >= 3.1E+09) and (frequency <= 5.3E+09):
                if speed == 0:
                    if foliage == 0:
                        if tx_height < 1.0:
                            n = 3.0374
                            sigma = 4.897
                        elif tx_height >= 1.0:
                            n = 2.606
                            sigma = 4.31
                        else:
                            print("Please specify a tx_height to use this model")
                            return 0,0
                    elif foliage ==1:
                        n = 2.7601
                        sigma = 4.8739
                if speed == 20:
                    if foliage == 0:
                        if tx_height < 1.0:
                            n = 2.961
                            sigma = 4.71
                        elif tx_height >= 1.0:
                            n = 2.667
                            sigma = 4.96
                        else:
                            print("Please specify a tx_height to use this model")
                            return 0,0
                    elif foliage ==1:
                        n = 2.8350
                        sigma = 5.3
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09], [3.1E+09,5.r [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "hilly_with_ridge":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.6
                sigma = 3.5
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09):
                n = 1.7
                sigma = 2.8
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09] or [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "dry_hilly":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.3
                sigma = 3.9
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09):
                n = 1.0
                sigma = 2.2
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09] or [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "very_mountainous":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.6
                sigma = 3.5
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09):
                n = 1.7
                sigma = 2.8
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09] or [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "fresh_water":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.9
                sigma = 3.8
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09):
                n = 1.9
                sigma = 3.1
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09] or [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "sea":
            if (frequency >= 9E+08) and (frequency <= 1.2E+09):
                n = 1.9
                sigma = 4.2
            elif (frequency >= 5.03E+09) and (frequency <= 5.091E+09):
                n = 1.5
                sigma = 2.6
            else:
                print("Please enter a frequency within [9E+08, 1.2E+09] or [5.03E+09, 5.091E+09].")
                return 0, 0
        elif scenario == "custom":
            if (n_custom != None) and (sigma_custom != None):    
                n = n_custom
                sigma = sigma_custom
            else:
                print("Please specify a n_custom and n_sigma for using custom feature.")
                return 0,0
        else:
            print("Using default values:", n, ", ", sigma)

        pl = pl_0 + 10 * n * math.log10(distance / self.d0)
        return pl, sigma
    
    def log_distance_alpha_beta_pl(self, distance, frequency, scenario="lightly_hilly_rural", rx_height=None, alpha_custom=None, beta_custom=None, sigma_custom=None):
        """
        Calculate Log-Distance alpha-beta (AB) Path Loss Model.

        Args:
            distance (float): Distance between transmitter and receiver (meters).
            frequency (float): Signal frequency (Hz).
            scenario (string): "lightly_hilly_rural".
            rx_height (float): UAV height, from 0 to 120 meters (higher values are approximated by the 120 m model).
            alpha_custom (float): alpha path loss exponent custom value, if scenario="custom".
            beta_custom (float): beta path loss term custom value, if scenario="custom".
            sigma_custom (float): path loss standard deviation custom value, if scenario="custom".

        Returns:
            float: Path Loss value in dB.
            float: Standard deviation value (if applicable).
        """
        wavelength = self.c / frequency
        
        alpha = 2.0
        beta = 35.3
        sigma = 3.4
        if scenario == "lightly_hilly_rural":
            if rx_height == None:
                print("Please specify a rx_height [0,120]. Using default values:", alpha,", ", beta, ", ", sigma)
            elif rx_height <= 5:
                alpha = 3.7
                beta = -1.3
                sigma = 7.7
                
            elif (rx_height > 5) and (rx_height <= 22.5):
                alpha = 2.9
                beta = 7.4
                sigma = 6.2
            elif (rx_height > 22.5) and (rx_height <= 45):
                alpha = 2.5
                beta = 20.4
                sigma = 5.2
            elif (rx_height > 45) and (rx_height <= 90):
                alpha = 2.1
                beta = 32.8
                sigma = 4.4
            elif rx_height > 90:
                alpha = 2.0
                beta = 35.3
                sigma = 3.4
            
        elif scenario == "custom":
            if (alpha_custom != None) and (beta_custom != None) and (sigma_custom != None):    
                alpha = alpha_custom
                beta = beta_custom
                sigma = sigma_custom
            else:
                print("Please specify a n_custom and n_sigma for using custom feature.")
                return 0,0
        else:
            print("Using default values:", alpha, ", ", beta,", ", sigma)

        pl = alpha * 10 * math.log10(distance) + beta
        return pl, sigma
    
    
    def two_ray_perfect_reflection_pl(self, distance, frequency, h_t = 1.5, h_r=30):
        """
        Calculate Two Ray Path Loss Model (assuming perfect reflection).

        Args:
            distance (float): Distance between transmitter and receiver (meters).
            frequency (float): Signal frequency (Hz)
            h_t (float): height of the transmitter (meters)
            h_r (float): height of the receiver (meters)

        Returns:
            float: Path Loss value in dB.
            float: Standard deviation value (not applicable) = 0.
        """
        wavelength = self.c / frequency
        
        sigma = 0.
        
        pl = -10 * math.log10((wavelength/(4*math.pi*distance))**2 * (2*math.sin((2*math.pi*h_t*h_r)/(wavelength*distance)))**2)
        return pl, sigma
    
    
    def modified_two_ray_pl(self, distance, frequency, h_t, h_r, G_l_custom=None, G_r_custom=None, R_custom=None, gamma_h_custom=None):
        """
        Calculate Modified Two Ray Path Loss Model based on three zones depending on transmitter and receiver height. 

        Args:
            distance (float): Ground distance between transmitter and receiver (meters).
            frequency (float): Signal frequency (Hz).
            h_t (float): height of the transmitter (meters).
            h_r (float): height of the receiver (meters).
            G_l_custom (float): direct path coefficient (optional).
            G_r_custom (float): reflected path coefficient (optional).
            R_custom (float): ground reflection coefficient (optional).
            gamma_h_custom (float): height-dependent propagation coefficient (optional).
        Returns:
            float: Path Loss value in dB.
            float: Standard deviation value (not applicable) = 0.
        """
        
        if (G_l_custom != None) and (G_r_custom != None) and (R_custom != None) and (gamma_h_custom != None):
            G_l = G_l_custom
            G_r = G_r_custom
            R = R_custom
            gamma_h = gamma_h_custom
        elif (G_l_custom == None) and (G_r_custom == None) and (R_custom == None) and (gamma_h_custom == None):
            if h_r > 60:
                G_l = 0
                G_r = 3.5
                gamma_h = 2
                R = -1
            elif (h_r <= 60) and (h_r > 30):
                G_l = 7
                G_r = 7
                gamma_h = 2.75
                R = -1
            elif h_r <= 30:
                G_l = 15
                G_r = 5
                gamma_h = 3.5
                R = -1
            else:
                print("Please specify a correct receiver height.")
                return 0,0
        else:
            print("Please specify a h_t and h_r, or alternatively all the custom values for G_l_custom, G_r_custom, R_custom, gamma_h_custom")
            return 0,0
            
        wavelength = self.c / frequency
        
        sigma = 0.
        
        l = distance*math.sqrt(1+((h_t-h_r)**2)/(distance**2))
        r_1 = math.sqrt(h_t**2 + ((distance*h_t)**2)/((h_t+h_r)**2))
        r_2 = math.sqrt(h_r**2 + (distance - (distance*h_t)/(h_t+h_r))**2)
        
        
        pl = 20 * math.log10((4 * math.pi)/wavelength) - 10 * gamma_h * math.log10(abs(G_l/l + R*G_r/(r_1+r_2)))
        
        return pl, sigma
    
    def dual_slope_pl(self, distance, frequency, gamma_1, gamma_2, default=None, d_b=9.0):
        """
        Calculate Dual Slope Path Loss Model.

        Args:
            distance (float): Distance between transmitter and receiver (meters).
            frequency (float): Signal frequency (Hz).
            d_b (float): Break distance, default to 9.0 (meters).
            gamma_1 (float): Path loss exponent 1.
            gamma_2 (float): Path loss exponent 2.
            default (int): default UAV height case. Only available: 20, 30 (meters).
        Returns:
            float: Path Loss value in dB.
            float: Zero-mean Gauss-distributed shadow fading in dB (psi).
        """
        wavelength = self.c / frequency
        
        psi = 0.
        
        pl_0  = 10.0 * math.log10(((4 * math.pi * self.d0)/wavelength)**2)
        
        if default == 20:
            gamma_1 = 0.74
            gamma_2 = 2.29
            psi = 5.5
        elif default == 30:
            gamma_1 = 0.74
            gamma_2 = 2.29
            psi = 3.9
        else:
            print("Using user provided gamma_1, gamma_2, psi=0")
        
        if distance < d_b:
            pl = pl_0 + 10 * gamma_1 * math.log10(distance/self.d0)
        elif distance >= d_b:
            pl = pl_0 + 10 * gamma_1 * math.log10(d_b/self.d0) + 10 * gamma_2 * math.log10(distance/self.d0)
        else:
            print("Please provide an adequate distance and break distance (m).")
            return 0,0
        return pl, psi
    
    def elevation_angle_pl(self, frequency, elev_angle, h_t, h_r):
        """
        Calculate elevation angle-based Path Loss Model.

        Args:
            frequency (float): Signal frequency (Hz)
            elev_angle (float): elevation angle between the UAV and the GS (degrees)
            h_t (float): height of the transmitter (meters)
            h_r (float): height of the receiver (meters)
        Returns:
            float: Path Loss value in dB.
            float: Standard deviation value (not applicable) = 0.
        """
        
        pl = 20 * math.log10((h_r-h_t)/math.sin(math.radians(elev_angle))) + 20 * math.log10(frequency*10**-6) - 27.55
        
        return pl, 0
    
    # Auxiliary methods for distances estimation
    def distance_3d(self, tx_position, rx_position):
        """
        Calculate 3D distance between transmitter and receiver positions.

        Args:
            tx_position (tuple): 3D position of transmitter (x, y, z) in meters.
            rx_position (tuple): 3D position of receiver (x, y, z) in meters.

        Returns:
            float: 3D distance between transmitter and receiver (meters).
        """
        dx = tx_position[0] - rx_position[0]
        dy = tx_position[1] - rx_position[1]
        dz = tx_position[2] - rx_position[2]
        distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        return distance_3d

    def distance_2d(self, tx_position, rx_position):
        """
        Calculate 2D distance between transmitter and receiver positions.

        Args:
            tx_position (tuple): 2D position of transmitter (x, y) in meters.
            rx_position (tuple): 2D position of receiver (x, y) in meters.

        Returns:
            float: 2D distance between transmitter and receiver (meters).
        """
        dx = tx_position[0] - rx_position[0]
        dy = tx_position[1] - rx_position[1]
        distance_2d = math.sqrt(dx**2 + dy**2)
        return distance_2d
    
    # Auxiliary methods for plotting 
    def create_path_loss_contour(self, tx_position, rx_position, range_x, range_y, points, pl_func='free_space_pl', **kwargs):
        """
        Create a path loss contour plot centered around the transmitter.

        Args:
            tx_position (tuple): Transmitter position (x, y) in meters.
            rx_position (tuple): Receiver position (x, y) in meters.
            range_x (float): range in meters from the tx_position[0]
            range_y (float): range in meters from the tx_position[1]
            points (int): number of points to simulate in both directions
            pl_func (str): "free_space_pl", "log_distance_pl", "log_distance_alpha_beta_pl", "two_ray_perfect_reflection_pl", "two_ray_non_perfect_reflection_pl", "dual_slope_pl", "elevation_angle_pl" 
        Returns:
            None
        """
        # Create a grid of coordinates centered around the transmitter
        x_range = np.linspace(tx_position[0] - range_x, tx_position[0] + range_x, points)  # Adjust the range and number of points as needed
        y_range = np.linspace(tx_position[1] - range_y, tx_position[1] + range_y, points)
        X, Y = np.meshgrid(x_range, y_range)

        # Calculate distances from transmitter for each point in the grid
        distances = np.sqrt((X - tx_position[0])**2 + (Y - tx_position[1])**2)


        pl_func_name = getattr(self, pl_func)
        
        # Vectorize the selected path loss function
        vectorized_pl_func = np.vectorize(pl_func_name, excluded=kwargs.keys())
    
        # Calculate path loss values for each distance using the selected path loss function
        path_losses, _ = vectorized_pl_func(distances, **kwargs)
        #path_losses, _ = pl_func_name(distances, **kwargs)

        # Create a contour plot
        plt.contourf(X, Y, path_losses, levels=20, cmap='viridis')
        plt.colorbar(label='Path Loss (dB)')

        # Add transmitter and receiver positions
        plt.scatter(tx_position[0], tx_position[1], label='Transmitter', color='blue', marker='x')
        plt.scatter(rx_position[0], rx_position[1], label='Receiver', color='red', marker='o')

        # Calculate the path loss value at the RX position
        rx_x, rx_y = rx_position
        rx_index_x = np.argmin(np.abs(x_range - rx_x))
        rx_index_y = np.argmin(np.abs(y_range - rx_y))
        rx_path_loss = path_losses[rx_index_y, rx_index_x]

        # Add a label with the RX position and path loss value
        plt.annotate(f'RX: ({rx_x}, {rx_y}) m\nPath Loss: {rx_path_loss:.2f} dB', 
                     xy=(rx_x, rx_y), xycoords='data',
                     xytext=(10, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        plt.annotate(f'TX: {tx_position} m', 
                     xy=(tx_position[0], tx_position[1]), xycoords='data',
                     xytext=(-60, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        
        # Set axis labels and legend
        plt.xlabel('X-coordinate (m)')
        plt.ylabel('Y-coordinate (m)')
        plt.legend()
        
        default_title = "Path loss contour plot: "
        
        # Show the plot
        plt.grid(True)
        plt.title(f'{default_title}{pl_func}')
        plt.show()



# Example usage:
if __name__ == "__main__":
    # Create a PathLossCalculator instance
    frequency = 2.4e9  # 2.4 GHz
    reference_distance = 1.0  # 1 meter
    n = 2.0  # Path Loss exponent for Log-Distance Path Loss Model
    calculator = PathLossCalculator(reference_distance, n)
    
    # Add transmitter and receiver positions
    tx_position = (10, 20)  # Replace with actual coordinates
    rx_position = (30, 40)  # Replace with actual coordinates
    
    # Create and display the path loss contour plot
    custom_params = {'frequency': 868000000}
    calculator.create_path_loss_contour(tx_position, rx_position, 100, 100, 100, pl_func='free_space_pl', **custom_params)
    calculator.create_path_loss_contour(tx_position, rx_position, 100, 100, 100, pl_func='log_distance_pl', **custom_params)
    

   

