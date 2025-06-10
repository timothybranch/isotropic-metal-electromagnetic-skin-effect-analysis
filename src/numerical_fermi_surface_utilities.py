import numpy as np
# from PyAstronomy.pyasl import generalizedESD
from scipy.constants import mu_0, pi, e, epsilon_0, hbar, elementary_charge
from scipy.optimize import root_scalar
import pyvista as pv
import json
from datetime import datetime
import os
import pandas as pd
from numba import jit, njit
from scipy.optimize import curve_fit
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

### Vector manipulation utilities

def angle_between_vectors(vec_1, vec_2):
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    vec_1 : ndarray
        The first vector
    vec_2 : ndarray
        The second vector

    Returns
    -------
    float
        The angle between the two vectors [rad]
    """

    cos_value = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

    if np.abs(cos_value-1) < 1e-10:
        cos_value = 1
    elif np.abs(cos_value+1) < 1e-10:
        cos_value = -1

    if cos_value > 1 or cos_value < -1:
        raise ValueError('The cosine value is out of bounds with a value of ' + str(cos_value) + ' for vectors ' + str(vec_1) + ' and ' + str(vec_2) + '.')

    return np.arccos(cos_value)

# calculate_angle_between_vectors = np.vectorize(calculate_angle_between_vectors, signature='(n),(n)->()')

def vector_projection(v, u):
    """
    Project vector v onto vector u.

    Parameters
    ----------
    v : ndarray
        The vector to be projected.
    u : ndarray
        The vector onto which v is projected.

    Returns
    -------
    ndarray
        The projection of vector v onto vector u.
    """
    return ((np.dot(v, u)) / (np.linalg.norm(u) ** 2)) * np.array(u)

def plane_projection(v, q, J):
    """
    Project vector v onto the plane defined by vectors q and J.

    Parameters
    ----------
    v : array_like
        The vector to be projected.
    q : array_like
        The first vector defining the plane.
    J : array_like
        The second vector defining the plane.

    Returns
    -------
    array_like
        The projection of vector v onto the plane defined by q and J.

    Raises
    ------
    ValueError
        If any of the input vectors do not have a length of 3.
    """
    if len(v) != 3:
        raise ValueError('Vector v with length 3 was expected.')
    if len(q) != 3:
        raise ValueError('Vector q with length 3 was expected.')
    if len(J) != 3:
        raise ValueError('Vector J with length 3 was expected.')
    plane_vec = np.cross(q, J)
    return v - ((np.dot(v, plane_vec)) / (np.linalg.norm(plane_vec) ** 2)) * plane_vec


def J_vec_calculator(angle, vec_1, vec_2):
    """
    Calculate the vector J based on the given angle and two input vectors.

    Parameters
    ----------
    angle : float
        The rotation angle [degrees]
    vec_1 : ndarray
        The first input vector 
    vec_2 : ndarray
        The second input vector

    Returns
    -------
    ndarray
        The calculated vector J
    """
    if angle % 180 == 0:
        return vec_1
    if angle % 180 == 90:
        return vec_2
    return np.array(vec_1) * np.cos(angle * pi / 180) + np.array(vec_2) * np.sin(angle * pi / 180)

@njit()
def cartesian_to_spherical(vector):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    vector : ndarray
        The vector in Cartesian coordinates.

    Returns
    -------
    array_like
        The spherical coordinates [r, theta, phi].
    """
    x = vector[0]
    y = vector[1]
    z = vector[2]

    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2

    r = np.sqrt(x_2 + y_2 + z_2)

    ## If vector is in +/i z-axis
    if x == 0 and y == 0:
        if z > 0:
            return np.array([r, 0., 0.])
        elif z < 0:
            return np.array([r, np.pi, 0.])
        elif z == 0:
            return np.array([0., 0., 0.])
    elif x == 0 and z == 0:
        if y > 0:
            return np.array([r, np.pi/2, np.pi/2])
        elif y < 0:
            return np.array([r, np.pi/2, 3*np.pi/2])
    elif y == 0 and z == 0:
        if x > 0:
            return np.array([r, np.pi/2, 0.])
        elif x < 0:
            return np.array([r, np.pi/2, np.pi])

    theta = np.arccos(z/r)
    phi = np.sign(y)*np.arccos(x/np.sqrt(x_2 + y_2))

    return np.array([r, theta, phi])

def spherical_to_cartesian(vector):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    vector : ndarray
        The vector in spherical coordinates [r, theta, phi].

    Returns
    -------
    array_like
        The Cartesian coordinates [x, y, z].
    """
    r = vector[0]
    theta = vector[1]
    phi = vector[2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([r, theta, phi])


##TODO: test this formatter
def generate_miller_latex_format(vector, type='symmetry equivalent'):
    """
    Generate LaTeX format for Miller indices.
    
    Parameters
    ----------
    vector : array_like
        A list or numpy array of the vector in [x, y, z] format.
    type : str, optional
        The type of vector to be represented. Default is 'symmetry equivalent', which gives the angle brackets notation.
        The other option is 'specific', which uses the square brackets notation.
    
    Returns
    -------
    str
        A LaTeX string representing the vector in Miller indices format.
    
    Raises
    ------
    ValueError
        If the type of vector representation is invalid.
    """

    miller_indices = vector_to_miller(vector)

    if type == 'symmetry equivalent':
        # Join the indices with angle brackets for LaTeX
        miller_latex_format = r'$\\langle{miller_indices[0]}{miller_indices[1]}{miller_indices[2]}\\rangle$'
    elif type == 'specific':
        # Join the indices with square brackets for LaTeX
        miller_latex_format = r'$\left[{miller_indices[0]}{miller_indices[1]}{miller_indices[2]}\right]$'
    else:
        raise ValueError('Invalid type of vector representation.')

    return miller_latex_format


##TODO: test if the latex bar fix works
def vector_to_miller(vector):
    """
    Convert a vector to Miller indices notation in LaTeX string format.
    
    Parameters
    ----------
    vector : array_like
        A list or numpy array of the vector in [x, y, z] format.
    
    Returns
    -------
    str
        A LaTeX string representing the vector in Miller indices notation.
    """
    # Normalize the vector so the smallest non-zero value (up to bit error) is 1
    min_non_zero = np.min(np.abs(vector[np.nonzero(vector)]))
    normalized_vec = np.round(vector / min_non_zero, 2)


    # Convert to Miller notation with overbars for negative values
    miller_indices = []
    for i, value in enumerate(normalized_vec):
        if value == 0:
            miller_indices[i] = '0'
        else:
            miller_index_string = f'{value:.2f}'.rstrip('0').rstrip('.')
            if value > 0:
                miller_indices[i] = miller_index_string
            else:
                miller_indices[i] = f'\\bar{{{miller_index_string}}}'
    
    return miller_indices


def find_closest_indices(values_list, target_values_list):
    """
    Find the indices in the list that correspond to the elements closest to the target values.

    Parameters
    ----------
    values_list : ndarray
        The list of values.
    target_values_list : ndarray
        The list of target values to find the closest elements for in the list.

    Returns
    -------
    ndarray
        A list of indices corresponding to the closest elements in the list for each target value.
    """

    closest_indices = []
    for target in target_values_list:
        ## For each target, find the closest value in the list, and then add the index to the results list
        differences = [abs(list_val - target) for list_val in values_list]
        
        closest_index = differences.index(min(differences))
        closest_indices.append(closest_index)
    return np.array(closest_indices)


##TODO: make this give the nicest vectors
def orthogonal_vectors(vector):
    """
    Calculate two orthogonal vectors to the given vector.

    Parameters
    ----------
    vector : ndarray
        The vector for which orthogonal vectors are to be calculated.

    Returns
    -------
    tuple of ndarray
        Two orthogonal vectors to the given vector.
    """

    if all(v == 0 for v in vector):
        raise ValueError("Zero vector does not have unique orthogonal vectors.")

    if vector[0] != 0 or vector[1] != 0:
        ortho_vec_1 = [-vector[1], vector[0], 0]
    else:
        ortho_vec_1 = [1, 0, 0]

    ortho_vec_2 = np.cross(vector, ortho_vec_1)

    return ortho_vec_1, ortho_vec_2

## Rotation functions
def rotation_matrix_x(phi_x):
    """
    Rotation matrix for a rotation around the x-axis (right-handed).

    Parameters
    ----------
    phi_x : float
        The rotation angle in degrees around the x-axis.

    Returns
    -------
    ndarray
        The rotation matrix for a rotation around the x-axis.
    """

    phi_x_rad = np.radians(phi_x % 360)
    special_cases = {
        0: np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]]),
        90: np.array([[1, 0, 0], 
                      [0, 0, -1], 
                      [0, 1, 0]]),
        180: np.array([[1, 0, 0], 
                       [0, -1, 0], 
                       [0, 0, -1]]),
        270: np.array([[1, 0, 0], 
                       [0, 0, 1], 
                       [0, -1, 0]])
    }
    
    if phi_x % 90 == 0:
        return special_cases[phi_x % 360]
    
    cos_phi = np.cos(phi_x_rad)
    sin_phi = np.sin(phi_x_rad)

    return np.array([
        [1, 0, 0],
        [0, cos_phi, -sin_phi],
        [0, sin_phi, cos_phi]
    ])


def rotation_matrix_y(phi_y):
    phi_y_rad = np.radians(phi_y % 360)
    special_cases = {
        0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        90: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        180: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        270: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    }
    if phi_y % 90 == 0:
        return special_cases[phi_y % 360]
    cos_phi = np.cos(phi_y_rad)
    sin_phi = np.sin(phi_y_rad)
    return np.array([
        [cos_phi, 0, sin_phi],
        [0, 1, 0],
        [-sin_phi, 0, cos_phi]
    ])


def rotation_matrix_z(phi_z):
    phi_z_rad = np.radians(phi_z % 360)
    special_cases = {
        0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        90: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        270: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    }
    if phi_z % 90 == 0:
        return special_cases[phi_z % 360]
    cos_phi = np.cos(phi_z_rad)
    sin_phi = np.sin(phi_z_rad)
    return np.array([
        [cos_phi, -sin_phi, 0],
        [sin_phi, cos_phi, 0],
        [0, 0, 1]
    ])


###TODO: add a initial 
def Rzyx_func(vector, phi_z, phi_y, phi_x):
    """
    Rotate a vector by angles around the positive z-axis, then y-axis, then x-axis.
    
    Parameters
    ----------
    vector : ndarray (3,)
        The vector to be rotated.
    phi_z : float
        The rotation angle in degrees around the positive z-axis.
    phi_y : float
        The rotation angle in degrees around the positive y-axis.
    phi_x : float
        The rotation angle in degrees around the positive x-axis.

    Returns
    -------
    ndarray
        The rotated vector.
    """
    Rz = rotation_matrix_z(phi_z)
    Ry = rotation_matrix_y(phi_y)
    Rx = rotation_matrix_x(phi_x)
    
    vector_rotated_z = np.matmul(Rz, vector)
    vector_rotated_zy = np.matmul(Ry, vector_rotated_z)
    vector_rotated_zyx = np.matmul(Rx, vector_rotated_zy)
    return vector_rotated_zyx

def Rxyz_func(vector_list, phi_x, phi_y, phi_z):
    """
    Rotate a vector by angles around the positive x-axis, then y-axis, then z-axis.
    
    Parameters
    ----------
    vector_list : ndarray (n,3)
        The list of vectors to be rotated.
    phi_x : float
        The rotation angle around the positive x-axis [degrees]
    phi_y : float
        The rotation angle around the positive y-axis [degrees]
    phi_z : float
        The rotation angle around the positive z-axis [degrees]

    Returns
    -------
    ndarray
        The rotated vector.
    """

    # Create rotation matrices
    Rz = rotation_matrix_z(phi_z)
    Ry = rotation_matrix_y(phi_y)
    Rx = rotation_matrix_x(phi_x)

    # Apply rotations to the entire array of vectors
    vector_list_rotated_xyz = np.dot(vector_list, Rx.T)
    vector_list_rotated_xyz = np.dot(vector_list_rotated_xyz, Ry.T)
    vector_list_rotated_xyz = np.dot(vector_list_rotated_xyz, Rz.T)

    return vector_list_rotated_xyz

Rxyz_func_vec = np.vectorize(Rxyz_func, excluded=['phi_x', 'phi_y', 'phi_z'])


##TODO implement this function that rotates vectors to the crystallographic directions
def rotate_to_crystallographic_directions(vectors, surface_normal_direction, surface_normal_misalignment=0, current_direction_angle=0):
    """
    Rotate a vector to the crystallographic directions.
    
    Options for crystallographic directions:
    - 'Primary' (<100>)
    - 'Secondary' (<111>)
    - 'Tertiary' (<110>)

    The default (0 degrees) current direction for each is:
    - 'Primary' : [0, 1, 0]
    - 'Secondary' : [1, 1, 0]
    - 'Tertiary' : [1, 0, 0]

    Parameters
    ----------
    vector : ndarray[3,]
        The vector to be rotated
    surface_normal_direction : string['Primary', 'Secondary', 'Tertiary']
        The direction of the surface normal
    surface_normal_misalignment : float
        The misalignment of the surface normal, around the current direction at 0 degrees [degrees]
    current_direction_angle : float
        The angle of the current direction [degrees]

    Returns
    -------
    ndarray[3,]
        The rotated vector
    """
    rotated_vectors = np.zeros_like(vectors)

    

    raise Exception('Not implemented')
    return rotated_vectors


def rotate_vector_towards(v1, v2, angle_degrees):
    """
    Rotate a vector towards another vector by an angle.
    
    Parameters
    ----------
    v1 : ndarray[3,]
        The vector to be rotated
    v2 : ndarray[3,]
        The vector to rotate towards
    angle_degrees : float
        The angle to rotate the vector by [degrees]

    Returns
    -------
    ndarray[3,]
        The rotated vector
    """
    angle_radians = np.radians(angle_degrees)

    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    axis_rotation = np.cross(v1, v2)
    axis_rotation_norm = axis_rotation / np.linalg.norm(axis_rotation)

    # Compute the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -axis_rotation_norm[2], axis_rotation_norm[1]],
                  [axis_rotation_norm[2], 0, -axis_rotation_norm[0]],
                  [-axis_rotation_norm[1], axis_rotation_norm[0], 0]])

    rotation_matrix = (np.eye(3) + 
                       sin_angle * K + 
                       (1 - cos_angle) * np.dot(K, K))

    # Rotate the vector
    rotated_vector = np.dot(rotation_matrix, v1)

    # new_vector = cos_angle * v1_norm + sin_angle * np.cross(v2_norm, v1_norm) + (1 - cos_angle) * np.dot(v2_norm, v1_norm) * v2_norm
    # rotated_vector = cos_angle * v1 + sin_angle * v2
    return rotated_vector


### Mesh utilities

def cartesian_vectors(k_length):
    """
    Generate wavevectors in the positive and negative Cartesian directions.
    
    Parameters
    ----------
    k_length : float
        Length of the wavevectors

    Returns
    -------
    ndarray
        Wavevectors in the positive and negative Cartesian directions.
    """
    
    unit_vectors = np.array([[1, 0, 0], 
                              [-1, 0, 0], 
                              [0, 1, 0], 
                              [0, -1, 0], 
                              [0, 0, 1], 
                              [0, 0, -1]])

    return k_length*unit_vectors


def extend_unit_vectors_to_fermi_wavevector(E_F, band_dispersion, k_F_length_initial_guess=1.0):
    """
    Calculate the Fermi wavevectors that are in the Cartesian axes, given a band dispersion relation and the Fermi energy.

    Parameters
    ----------
    E_F : float
        Fermi energy in eV
    band_dispersion : function
        Band dispersion relation
    k_F_length_initial_guess : float, optional
        Initial guess for the Fermi wavevector length (in Angstroms)

    Returns
    -------
    wavevector : ndarray([6,3])
        Fermi wavevector in the Cartesian axes.

    """

    def fermi_surface_equation_x(k_F_length):
        k_F_vector = k_F_length * np.array([1, 0, 0])
        return band_dispersion(np.array([k_F_vector])) - E_F

    def fermi_surface_equation_y(k_F_length):
        k_F_vector = k_F_length * np.array([0, 1, 0])
        return band_dispersion(np.array([k_F_vector])) - E_F

    def fermi_surface_equation_z(k_F_length):
        k_F_vector = k_F_length * np.array([0, 0, 1])
        return band_dispersion(np.array([k_F_vector])) - E_F

    # Using root_scalar with bounds
    result_x = root_scalar(fermi_surface_equation_x, bracket=[0., 10 * k_F_length_initial_guess], method='brentq')
    result_y = root_scalar(fermi_surface_equation_y, bracket=[0., 10 * k_F_length_initial_guess], method='brentq')
    result_z = root_scalar(fermi_surface_equation_z, bracket=[0., 10 * k_F_length_initial_guess], method='brentq')

    k_F_length_solution_x = np.abs(result_x.root)
    k_F_length_solution_y = np.abs(result_y.root)
    k_F_length_solution_z = np.abs(result_z.root)

    k_F_x_pos = k_F_length_solution_x * np.array([1, 0, 0])
    k_F_y_pos = k_F_length_solution_y * np.array([0, 1, 0])
    k_F_z_pos = k_F_length_solution_z * np.array([0, 0, 1])

    k_F_x_neg = k_F_length_solution_x * np.array([-1, 0, 0])
    k_F_y_neg = k_F_length_solution_y * np.array([0, -1, 0])
    k_F_z_neg = k_F_length_solution_z * np.array([0, 0, -1])

    return np.array([k_F_z_pos, k_F_x_pos, k_F_y_pos, k_F_x_neg, k_F_y_neg, k_F_z_neg])


def extend_spherical_to_fermi_wavevector(E_F, band_dispersion, r, theta, phi):
    """
    Convert spherical coordinates to a Fermi wavevector k_F, given a band dispersion relation and the Fermi energy.

    Parameters
    ----------
    E_F : float
        Fermi energy in eV
    band_dispersion : function
        Band dispersion relation
    r : float
        Radius of the sphere
    theta : float
        Theta angle in radians
    phi : float
        Phi angle in radians   

    Returns
    -------
    wavevector : np.array
        Fermi wavevector in Cartesian coordinates, in the same direction but at the Fermi surface.   

    """
    # @njit() Did not help
    def angles_to_cartesian(theta, phi):
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # @jit() # Did not work
    def fermi_surface_equation(k_F_length):
        # k_F_vector = k_F_length * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        k_F_vector = k_F_length * angles_to_cartesian(theta, phi)
        return band_dispersion(np.array([k_F_vector])) - E_F

    # Improved initial guess (example: based on some physical insight or empirical data)
    k_F_length_initial_guess = max(r, 0.1)  # Ensure it's positive and non-zero

    # Using root_scalar with bounds
    result = root_scalar(fermi_surface_equation, bracket=[0.01, 10 * k_F_length_initial_guess], method='brentq')

    k_F_length_solution = result.root
    k_F = k_F_length_solution * angles_to_cartesian(theta, phi)
    return k_F

## Mesh generation functions

def generate_mesh_from_fermi_surface_data(verts, face_indices):
    """
    Generate a mesh from Fermi surface data.

    Parameters
    ----------
    verts : ndarray([n,3])
        Vertices of the mesh
    face_indices : ndarray([n,3])
        Faces of the mesh

    Returns
    -------
    pv.PolyData
        Mesh
    """

    # Add 3's to the face_indices data so it works with pv
    face_indices_with_3s = np.hstack([np.full((face_indices.shape[0], 1), 3), face_indices])

    return pv.PolyData(verts, face_indices_with_3s)
    

def subdivide_mesh(mesh, E_F, band_dispersion, extend_to_FS=True):
    """
    Subdivide a mesh by adding midpoints to each edge and creating new faces.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to subdivide
    extend_to_FS : bool, optional
        Whether to extend the mesh to the Fermi surface

    Returns
    -------
    pv.PolyData
        Subdivided mesh
    """
    # Get vertices and faces
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    
    # Create new vertices at the midpoints of each edge
    new_vertices = []
    edge_to_mid = {}
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge not in edge_to_mid:
                mid = (vertices[edge[0]] + vertices[edge[1]]) / 2
                edge_to_mid[edge] = len(vertices) + len(new_vertices)

                if extend_to_FS:
                    r, theta, phi = tuple(cartesian_to_spherical(mid))
                    mid_extended = extend_spherical_to_fermi_wavevector(E_F, band_dispersion, r, theta, phi)
                    new_vertices.append(mid_extended)
                else:
                    new_vertices.append(mid)
                
    
    # Combine old and new vertices
    vertices = np.vstack([vertices, new_vertices])
    
    # Create new faces
    new_faces = []
    for face in faces:
        vert1, vert2, vert3 = face
        midvert1 = edge_to_mid[tuple(sorted([vert1, vert2]))]
        midvert2 = edge_to_mid[tuple(sorted([vert2, vert3]))]
        midvert3 = edge_to_mid[tuple(sorted([vert3, vert1]))]

        # _, theta1, phi1 = tuple(fs.cart2sph(vertices[midvert1]))
        # _, theta2, phi2 = tuple(fs.cart2sph(vertices[midvert2]))
        # _, theta3, phi3 = tuple(fs.cart2sph(vertices[midvert3]))

        # surfvert1 = direction_to_fermi_wavevector(E_f, band_dispersion, theta1, phi1)
        # surfvert2 = direction_to_fermi_wavevector(E_f, band_dispersion, theta2, phi2)
        # surfvert3 = direction_to_fermi_wavevector(E_f, band_dispersion, theta3, phi3)

        new_faces.extend([
            [vert1, midvert1, midvert3],
            [midvert1, vert2, midvert2],
            [midvert3, midvert2, vert3],
            [midvert1, midvert2, midvert3]
        ])
    
    # Create new mesh
    array_of_threes = np.full((len(new_faces), 1), 3)
    face_array = np.hstack([array_of_threes, new_faces]).astype(np.int32).flatten()

    new_mesh = pv.PolyData(vertices, face_array)
    return new_mesh

## Mesh element calculation functions

def mesh_triangle_area(vert1, vert2, vert3):
    """
    Calculate the area of a triangle given its vertices.

    Parameters
    ----------
    vert1 : ndarray([3])
        First vertex of the triangle
    vert2 : ndarray([3])
        Second vertex of the triangle
    vert3 : ndarray([3])
        Third vertex of the triangle

    Returns
    -------
    float
        Area of the triangle
    """
    return 0.5 * np.linalg.norm(np.cross(vert2 - vert1, vert3 - vert1))


def mesh_triangle_center(vert1, vert2, vert3):
    """
    Calculate the center of a triangle given its vertices.

    Parameters
    ----------
    vert1 : ndarray([3])
        First vertex of the triangle
    vert2 : ndarray([3])
        Second vertex of the triangle
    vert3 : ndarray([3])
        Third vertex of the triangle

    Returns
    -------
    ndarray([3])
        Center of the triangle
    """
    return (vert1 + vert2 + vert3) / 3


def mesh_triangle_normal(vert1, vert2, vert3):
    """
    Calculate the normal of a triangle given its vertices.

    Parameters
    ----------
    vert1 : ndarray([3])
        First vertex of the triangle
    vert2 : ndarray([3])
        Second vertex of the triangle
    vert3 : ndarray([3])
        Third vertex of the triangle

    Returns
    -------
    ndarray([3])
        Normal of the triangle
    """

    cross_product = np.cross(vert2 - vert1, vert3 - vert1)
    cross_product_norm = np.linalg.norm(cross_product)

    return cross_product / cross_product_norm

def mesh_get_edge_indices(mesh):
    """
    Return the edge indices for a pyvista mesh in a more usable format.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to get the edge indices from

    Returns
    -------
    ndarray([n,2])
        Edge indices for the mesh, where n is the number of edges, and the row index is the edge index
    """

    faces = mesh_get_face_indices(mesh)
    n_faces = len(faces)

    edges = np.empty((3*n_faces,2))

    for i, face in enumerate(faces):
        edge1 = np.sort([face[0],face[1]])
        edge2 = np.sort([face[0],face[2]])
        edge3 = np.sort([face[1],face[2]])

        edges[3*i] = edge1
        edges[3*i+1] = edge2
        edges[3*i+2] = edge3

    # Remove duplicates
    edges = np.unique(edges, axis=0).astype(int)

    return edges

def mesh_get_face_indices(mesh):
    """
    Return the vertex indices for a pyvista mesh in a more usable format.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to get the face indices from

    Returns
    -------
    ndarray([n,3])
        Vertex indices for the mesh, where n is the number of faces, and the row index is the face index
    """
    return mesh.faces.reshape(-1, 4)[:, 1:]

def mesh_get_vertex_areas(mesh):
    """
    Return the areas of mesh vertices.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to get the vertex areas from

    Returns
    -------
    ndarray([n])
        Areas of the mesh vertices
    """
    
    faces = mesh_get_face_indices(mesh)
    n_verts = mesh.n_points
    vertex_areas = np.zeros(n_verts)

    for face in faces:
        area = mesh_triangle_area(mesh.points[face[0]], mesh.points[face[1]], mesh.points[face[2]])
        for vertex in face:
            vertex_areas[vertex] += area / 3

    return vertex_areas

def mesh_calculate_elementwise_properties(mesh):
    """
    Calculate the properties of a mesh elementwise.

    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to calculate the properties of (n faces, all triangles)

    Returns
    -------
    mesh_element_areas : ndarray([n])
        Areas of the mesh elements
    mesh_element_centers : ndarray([n,3])
        Centers of the mesh elements
    mesh_element_normals : ndarray([n,3])
        Normals of the mesh elements
    """

    if not mesh.is_all_triangles:
        raise ValueError('Mesh is not all triangles')

    verts = mesh.points
    faces = mesh_get_face_indices(mesh)

    mesh_element_areas = np.zeros(len(faces))
    mesh_element_centers = np.zeros((len(faces), 3))
    mesh_element_normals = np.zeros((len(faces), 3))

    for i, face in enumerate(faces):

        vert1 = verts[face[0]]
        vert2 = verts[face[1]]
        vert3 = verts[face[2]]

        mesh_element_areas[i] = mesh_triangle_area(vert1, vert2, vert3)
        mesh_element_centers[i] = mesh_triangle_center(vert1, vert2, vert3)
        mesh_element_normals[i] = mesh_triangle_normal(vert1, vert2, vert3)

    return mesh_element_areas, mesh_element_centers, mesh_element_normals

def fold_points_into_first_Brillouin_zone(points : NDArray[np.float64], Brillouin_zone_edge_length : float) -> NDArray[np.float64]:
    """
    Fold the k_F vectors into the first Brillouin zone.

    Parameters
    ----------
    points : ndarray(float) of shape (n,3)
        Points to fold
    Brillouin_zone_edge_length : float
        Edge length of the Brillouin zone

    Returns
    -------
    ndarray(float) of shape (n,3)
        Points folded into the first Brillouin zone.
        If all points are within +/- Brillouin_zone_edge_length in all dimensions, returns the original points.
    """

    assert points.shape[1] == 3, "Points must be a 3D array"
    assert Brillouin_zone_edge_length > 0, "Brillouin zone edge length must be positive"

    # Check if all points lie within +/- Brillouin_zone_edge_length in all dimensions
    within_bounds = np.all(
        (points >= -Brillouin_zone_edge_length) & 
        (points <= Brillouin_zone_edge_length)
    )
    if within_bounds:
        return points


    points_folded = points.copy()
    for i in [0,1,2]:
        ## Move into the positive octant
        points_folded[:,i] = np.abs(points_folded[:,i])
        points_folded[:,i] = (points_folded[:,i] + Brillouin_zone_edge_length) % (2*Brillouin_zone_edge_length) - Brillouin_zone_edge_length

    return points_folded

# @njit() # Did not work
def mesh_interpolate_Fermi_velocity(mesh, v_Fs):
    """
    Interpolate the Fermi velocity to the mesh vertices for a mesh.
    
    Parameters
    ----------
    mesh : pv.PolyData
        Mesh to interpolate the Fermi velocity to
    v_Fs : ndarray([m,3])
        Fermi velocities (v_F_x, v_F_y, v_F_z) of the mesh vertices

    Returns
    -------
    v_F : ndarray([n,3])
        Fermi velocitites (v_F_x, v_F_y, v_F_z) of the mesh centers
    mesh_element_areas : ndarray([n])
        Areas of the mesh elements
    mesh_element_centers : ndarray([n,3])
        Centers of the mesh elements
    mesh_element_normals : ndarray([n,3])
        Normals of the mesh elements
    
    Notes
    -----
    m : int
        Number of vertices in the mesh.
    n : int
        Number of faces in the mesh.
    """

    vertices = mesh.points
    faces = mesh_get_face_indices(mesh)

    mesh_element_areas, mesh_element_centers, mesh_element_normals = mesh_calculate_elementwise_properties(mesh)

    # Initialize array to store interpolated Fermi velocities
    v_F_centers = np.zeros_like(mesh_element_centers)

    for i, face in enumerate(faces):
        # Get the vertices of the current face
        triangle_vertices = vertices[face]
        
        # Get the Fermi velocities at the vertices of the current face
        triangle_v_Fs = v_Fs[face]
        
        # Calculate barycentric coordinates
        v0 = triangle_vertices[1] - triangle_vertices[0]
        v1 = triangle_vertices[2] - triangle_vertices[0]
        v2 = mesh_element_centers[i] - triangle_vertices[0]
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        # Interpolate Fermi velocity
        v_F_centers[i] = u * triangle_v_Fs[0] + v * triangle_v_Fs[1] + w * triangle_v_Fs[2]

    return v_F_centers, mesh_element_areas, mesh_element_centers, mesh_element_normals

## Mesh data file generation

def generate_Fermi_surface_mesh_face_data(mesh, v_Fs):
    """
    Generate a file with the Fermi surface properties for each face of the mesh.

    Parameters
    ----------
    mesh : pv.PolyData [Angstrom]
        Mesh to generate the Fermi surface mesh file for
    v_Fs : ndarray([m,3]) [m/s]
        Fermi velocities of the mesh vertices

    Returns
    -------
    Dataframe with info (n rows)
        Face center positions (k_x, k_y, k_z) [1/Angstrom]
        Fermi velocity at face center (v_x, v_y, v_z) [m/s]
        Face area [1/Angstrom^2]
    
    Notes
    -----
    m : int
        Number of vertices in the mesh
    n : int
        Number of faces in the mesh
    """

    v_F_face, area_face, face_center, mesh_normals = mesh_interpolate_Fermi_velocity(mesh, v_Fs)

    # Flatten the face_center and v_F_face arrays into separate columns
    face_center_x, face_center_y, face_center_z = face_center[:, 0], face_center[:, 1], face_center[:, 2]
    v_F_face_x, v_F_face_y, v_F_face_z = v_F_face[:, 0], v_F_face[:, 1], v_F_face[:, 2]
    mesh_normal_x, mesh_normal_y, mesh_normal_z = mesh_normals[:, 0], mesh_normals[:, 1], mesh_normals[:, 2]

    dataframe_face_data = pd.DataFrame({
        'k_F_x [1/Angstrom]': face_center_x,
        'k_F_y [1/Angstrom]': face_center_y,
        'k_F_z [1/Angstrom]': face_center_z,
        'k_F_abs [1/Angstrom]': np.linalg.norm(face_center, axis=1),
        'v_F_x [m/s]': v_F_face_x,
        'v_F_y [m/s]': v_F_face_y,
        'v_F_z [m/s]': v_F_face_z,
        'v_F_abs [m/s]': np.linalg.norm(v_F_face, axis=1),
        'Weights [1/Angstrom^2]': area_face,
        'Normal_x': mesh_normal_x,
        'Normal_y': mesh_normal_y,
        'Normal_z': mesh_normal_z,
    })

    return dataframe_face_data

def generate_Fermi_surface_mesh_vertex_data(mesh, v_Fs):
    """
    Generate a file with the Fermi surface properties for each vertex of the mesh.

    Parameters
    ----------
    mesh : pv.PolyData [Angstrom]
        Mesh to generate the Fermi surface mesh file for
    v_Fs : ndarray([n]) [m/s]
        Fermi velocities of the mesh vertices

    Returns
    -------
    Dataframe with info
        Face center positions (k_x, k_y, k_z) [1/Angstrom]
        Fermi velocity at face center (v_x, v_y, v_z) [m/s]
        Face area [1/Angstrom^2]
    """

    verts = mesh.points

    vert_areas = mesh_get_vertex_areas(mesh)

    # Flatten the verts and v_Fs arrays into separate columns
    verts_x, verts_y, verts_z = verts[:, 0], verts[:, 1], verts[:, 2]
    v_F_x, v_F_y, v_F_z = v_Fs[:, 0], v_Fs[:, 1], v_Fs[:, 2]

    dataframe_vert_data = pd.DataFrame({
        'k_F_x [1/Angstrom]': verts_x,
        'k_F_y [1/Angstrom]': verts_y,
        'k_F_z [1/Angstrom]': verts_z,
        'k_F_abs [1/Angstrom]': np.linalg.norm(verts, axis=1),
        'v_F_x [m/s]': v_F_x,
        'v_F_y [m/s]': v_F_y,
        'v_F_z [m/s]': v_F_z,
        'v_F_abs [m/s]': np.linalg.norm(v_Fs, axis=1),
        'Weights [1/Angstrom^2]': vert_areas,
    })

    return dataframe_vert_data

def generate_mesh_information_file(mesh_name, mesh_directory, mesh, band_dispersion_model_info, mesh_iterations, mesh_starting_point='octahedron', mesh_iteration_type='edge bisection', notes=''):
    """
    Generate a file with information about the mesh.

    Parameters
    ----------
    mesh_name : str
        Name of the mesh
    mesh_directory : str
        Directory to save the mesh information file to
    mesh : pv.PolyData
        Mesh to generate the information file for
    band_dispersion_model_info : dict
        Band dispersion relation name and parameters
    mesh_interations : int
        Number of mesh interations
    mesh_starting_point : str
        Starting point for the mesh generation
    mesh_interation_type : str
        Type of mesh iteration

    Returns
    -------
    None
    """
    
    info_filename = mesh_name + "_info.json"
    info_filepath = os.path.join(mesh_directory, info_filename)

    verts = mesh.points
    num_verts = verts.shape[0]
    faces = mesh_get_face_indices(mesh)
    num_faces = faces.shape[0]
    edges = mesh_get_edge_indices(mesh)
    num_edges = edges.shape[0]

    mesh_volume = mesh.volume
    mesh_surface_area = mesh.area

    model_parameters = band_dispersion_model_info
    
    mesh_generation_properties = {
        "mesh_starting_point": mesh_starting_point,
        "mesh_iteration_type": mesh_iteration_type,
        "mesh_iterations": mesh_iterations
    }

    mesh_properties = {
        "num_vertices": num_verts,
        "num_faces": num_faces,
        "num_edges": num_edges,
        "mesh_volume": mesh_volume,
        "mesh_surface_area": mesh_surface_area,
    }

    mesh_info = {
        "mesh_name": mesh_name,
        "generation_date": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "model_parameters": model_parameters,
        "mesh_generation_properties": mesh_generation_properties,
        "mesh_properties": mesh_properties,
        "notes": notes
    }

    if os.path.exists(info_filepath):
        print(f'The file {info_filepath} already exists. Not saving.')
    else:
        with open(info_filepath, 'w') as f:
            json.dump(mesh_info, f, indent=4)


### Mesh information and data file loading

#TODO remove the csv file loading into dataframe to save memory and time
def load_all_mesh_information(mesh_directory):
    """
    Load all Fermi surface mesh information files from a directory.

    Parameters
    ----------
    mesh_directory : str
        Directory to load the mesh data from

    Returns
    -------
    FS_models : list[FS_model_name]
        List of Fermi surface models
    FS_model_levels : dict([FS_model_name] list[level_number])
        Dictionary with Fermi surface models and their levels
    FS_mesh_information : dict([mesh_name] json)
        Dictionary with mesh information
    FS_mesh_filepaths : dict([mesh_name] str)
        Dictionary with mesh vtk filepaths
    FS_face_data_filepaths : dict([mesh_name] str)
        Dictionary with face data csv filepaths
    """

    ## Initialize the dictionaries
    FS_mesh_information = {}
    FS_mesh_data = {}
    FS_model_levels = {}
    FS_face_data_filepaths = {}
    FS_mesh_filepaths = {}

    # Look through the folder "mesh_directory" and find every .json file
    for root, dirs, files in os.walk(mesh_directory):
        for filename in files:
            
            ## Only look at the json files
            if filename.endswith(".json"):
                json_filepath = os.path.join(root, filename)
                
                # Load the json file
                with open(json_filepath, 'r') as f:
                    mesh_information = json.load(f)
                
                # Use the key as "mesh_name" from the .json
                mesh_name = mesh_information["mesh_name"]
                
                # Save the .json data to FS_mesh_information
                FS_mesh_information[mesh_name] = mesh_information
                
                csv_filename = f"{mesh_name}_face_data.csv"
                csv_filepath = os.path.join(root, csv_filename)
                FS_face_data_filepaths[mesh_name] = csv_filepath

                mesh_filename = f"{mesh_name}.vtk"
                mesh_filepath = os.path.join(root, mesh_filename)
                FS_mesh_filepaths[mesh_name] = mesh_filepath

    for mesh_name in FS_mesh_information.keys():
        # Extract the common mesh name by removing the "_Level-n" part
        FS_model_name = mesh_name.split('_Level-')[0]
        
        # Initialize the list of levels if the common name is not already in the dictionary
        if FS_model_name not in FS_model_levels:
            FS_model_levels[FS_model_name] = []
        
        # Extract the level number and add it to the list of levels for the common name
        # level_number = int(mesh_name.split('_Level-')[1])

        ## The levels '4_wannierized' don't work for casting as int
        level_number = (mesh_name.split('_Level-')[1])
        FS_model_levels[FS_model_name].append(level_number)
    
    ## If it's a basic FS model with only integer level numbers, sort them as integers
    ## Skip if it's the ReO3 FS mesh with the wannierized levels
    for model_name in FS_model_levels:
        try:
            FS_model_levels[model_name].sort(key=lambda x: int(x))
        except ValueError:
            pass


    FS_models = list(FS_model_levels.keys())

    return FS_models, FS_model_levels, FS_mesh_information, FS_mesh_filepaths, FS_face_data_filepaths

def load_selected_mesh_data(mesh_directory, FS_mesh_filepaths):
    """
    Load all Fermi surface mesh information files and data from a directory.

    Parameters
    ----------
    mesh_directory : str
        Directory to load the mesh data from
    FS_mesh_filepaths : dict([mesh_name] str)
        Dictionary with mesh filepaths

    Returns
    -------
    FS_mesh_data : dict([mesh_name] dataframe)
        Dictionary with mesh data for the selected Fermi surface meshes
    """

    FS_mesh_data = {}

    for mesh_name, FS_mesh_filepath in FS_mesh_filepaths.items():
        if os.path.exists(FS_mesh_filepath):
            df = pd.read_csv(FS_mesh_filepath)
            # Save the dataframe to FS_mesh_data dictionary using the same key
            FS_mesh_data[mesh_name] = df

    return FS_mesh_data


def load_all_mesh_data(mesh_directory):
    """
    Load all Fermi surface mesh information files and data from a directory.

    Parameters
    ----------
    mesh_directory : str
        Directory to load the mesh data from

    Returns
    -------
    FS_models : list[FS_model_name]
        List of Fermi surface models
    FS_model_levels : dict([FS_model_name] list[level_number])
        Dictionary with Fermi surface models and their levels
    FS_mesh_information : dict([mesh_name] json)
        Dictionary with mesh information
    FS_mesh_data : dict([mesh_name] dataframe)
        Dictionary with mesh data
    """

    FS_mesh_information = {}
    FS_mesh_data = {}
    FS_model_levels = {}

    # Look through the folder "mesh_directory" and find every .json file
    for root, dirs, files in os.walk(mesh_directory):
        for filename in files:
            if filename.endswith(".json"):
                json_filepath = os.path.join(root, filename)
                
                # Load the json file
                with open(json_filepath, 'r') as f:
                    mesh_information = json.load(f)
                
                # Use the key as "mesh_name" from the .json
                mesh_name = mesh_information["mesh_name"]
                
                # Save the .json data to FS_mesh_information
                FS_mesh_information[mesh_name] = mesh_information
                
                ## Using the face data, not the vertex data
                csv_filename = f"{mesh_name}_face_data.csv"
                csv_filepath = os.path.join(root, csv_filename)
                
                if os.path.exists(csv_filepath):
                    df = pd.read_csv(csv_filepath)
                    # Save the dataframe to FS_mesh_data dictionary using the same key
                    FS_mesh_data[mesh_name] = df

    for mesh_name in FS_mesh_information.keys():
        # Extract the common mesh name by removing the "_Level-n" part
        FS_model_name = mesh_name.split('_Level-')[0]
        
        # Initialize the list of levels if the common name is not already in the dictionary
        if FS_model_name not in FS_model_levels:
            FS_model_levels[FS_model_name] = []
        
        # Extract the level number and add it to the list of levels for the common name
        level_number = int(mesh_name.split('_Level-')[1])
        FS_model_levels[FS_model_name].append(level_number)
    
    for model_name in FS_model_levels:
        FS_model_levels[model_name].sort()

    FS_models = list(FS_model_levels.keys())

    return FS_models, FS_model_levels, FS_mesh_information, FS_mesh_data


def FS_mesh_name(FS_model_name, FS_level):
    """
    Create a mesh name for a Fermi surface mesh.

    Parameters
    ----------
    FS_model_name : str
        Name of the Fermi surface model
    FS_level : int
        Level of the Fermi surface mesh

    Returns
    -------
    str
        Name of the Fermi surface mesh
    """

    return FS_model_name + '_Level-' + str(FS_level)


def load_FS_mesh_dataframe_to_arrays(mesh_dataframe : pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load a Fermi surface mesh dataframe (face-based) to arrays.

    Parameters
    ----------
    mesh_dataframe : pd.DataFrame
        Dataframe to load the mesh data from

    

    Returns
    -------
    positions : ndarray([n,3])
        Positions of the mesh face centers
    v_Fs : ndarray([n,3])
        Fermi velocities of the mesh face centers
    normals : ndarray([n,3])
        Normals of the mesh faces
    weights : ndarray([n])
        Weights of the mesh faces
    normal_vF_angles : ndarray([n,3])
        Angles between the normal and the v_F
    normal_vF_angle_avg : ndarray([n])
        Average angle between the normal and the v_F

    Notes
    -----
    n : int
        Number of faces
    """

    positions = np.array(mesh_dataframe[['k_F_x [1/Angstrom]', 'k_F_y [1/Angstrom]', 'k_F_z [1/Angstrom]']].values, dtype=float)
    v_Fs = np.array(mesh_dataframe[['v_F_x [m/s]', 'v_F_y [m/s]', 'v_F_z [m/s]']].values, dtype=float)
    try:
        normals = np.array(mesh_dataframe[['Normal_x', 'Normal_y', 'Normal_z']].values, dtype=float)
    except KeyError:
        normals = np.array([])

    weights = np.array(mesh_dataframe['Weights [1/Angstrom^2]'].values, dtype=float)

    try:
        # normal_vF_angles = np.array(mesh_dataframe[['Normal_vF_angle_1 [rad]', 'Normal_vF_angle_2 [rad]', 'Normal_vF_angle_3 [rad]']].values, dtype=float)
        normal_vF_angle_avg = np.array(mesh_dataframe['Normal_vF_angle_avg [rad]'].values, dtype=float)
    except KeyError:
        # normal_vF_angles = np.array([])
        normal_vF_angle_avg = np.array([])

    return positions, v_Fs, normals, weights, normal_vF_angle_avg


### Electronic transport property calculation functions

def calculate_average_Fermi_velocity(v_F_abs, weights):
    """
    Calculate the average Fermi velocity of a mesh.

    Parameters
    ----------
    v_F_abs : ndarray([n]) [m/s]
        Absolute Fermi velocities of the mesh vertices
    weights : ndarray([n]) [1/Angstrom^2]
        Weights of the mesh vertices

    Returns
    -------
    float [m/s]
        Average Fermi velocity of the mesh
    """

    v_F_avg = np.average(v_F_abs, weights=weights)

    return v_F_avg


def calculate_average_Fermi_velocity_RMS(v_F_abs, weights):
    """
    Calculate the root-mean-square Fermi velocity of a mesh.

    Parameters
    ----------
    v_F_abs : ndarray([n]) [m/s]
        Absolute Fermi velocities of the mesh vertices
    weights : ndarray([n]) [1/Angstrom^2]
        Weights of the mesh vertices

    Returns
    -------
    float [m/s]
        Root-mean-square Fermi velocity of the mesh
    """

    v_F_rms = np.sqrt(np.average(v_F_abs**2, weights=weights))

    return v_F_rms

def calculate_average_Fermi_velocity_harmonic(v_F_abs, weights):
    """
    Calculate the harmonic average (reciprocal of average of reciprocal) Fermi velocity of a mesh.

    Parameters
    ----------
    v_F_abs : ndarray([n]) [m/s]
        Absolute Fermi velocities of the mesh vertices
    weights : ndarray([n]) [1/Angstrom^2]
        Weights of the mesh vertices

    Returns
    -------
    float [m/s]
        Harmonic average Fermi velocity of the mesh
    """

    v_F_harmonic = np.power(np.average(v_F_abs**-1, weights=weights), -1)

    return v_F_harmonic

def calculate_plasma_frequency_from_Fermi_velocity(v_F_magnitudes : NDArray[np.float64], weights : NDArray[np.float64]):
    """
    Calculate the plasma frequency from the Fermi velocity for a mesh

    Parameters
    ----------
    v_F_magnitudes : NDArray[np.float64] of length n [m/s]
        Fermi velocity magnitudes of the mesh vertices
    weights : NDArray[np.float64] of length n [1/m^2]
        Weights of the mesh vertices

    Returns
    -------
    float
        Plasma frequency of the mesh
    """
    plasma_frequency = np.sqrt((elementary_charge**2)/(12*pi**3*hbar*epsilon_0)*np.sum(v_F_magnitudes*weights))
    return plasma_frequency

##TODO change this
def calculate_charge_carrier_density_ReO3(k_F_vectors : NDArray[np.float64], weights : NDArray[np.float64], Brillouin_zone_edge_length : float):
    """
    Calculate the charge carrier density of a mesh of ReO3.

    Adapted to work on the inverted gamma Fermi surface sheet.

    Cut off all aspects outside of a box with dimensions [0, Brillouin_zone_edge_length]^3.

    Parameters
    ----------
    k_F_magnitudes : NDArray[np.float64] of length n [1/m]
        Fermi velocity magnitudes of the mesh vertices 
    weights : NDArray[np.float64] of length n [1/m^2]
        Weights of the mesh vertices
    Brillouin_zone_edge_length : float [1/m]
        Edge length of the Brillouin zone

    Returns
    -------
    float
        Charge carrier density of the mesh [1/m^3]
    """

    ## Fold the k_F points into the first Brillouin zone
    k_F_folded = k_F_vectors.copy()
    for i in [0,1,2]:
        ## Make sure all points are in (0, 2* BZ_edge)^3, then shift to center at the origin
        k_F_folded[:,i] = (k_F_folded[:,i] % 2*Brillouin_zone_edge_length) - Brillouin_zone_edge_length

    # Filter out k-points that exceed the Brillouin zone edge
    # points_in_octant = ~(np.max(np.abs(k_F_vectors_adjusted), axis=1) > Brillouin_zone_edge_length)
    # filtered_k_F_vectors = k_F_vectors_adjusted[points_in_octant]
    # filtered_weights = weights[points_in_octant]
    n_ccs = (1/(12*pi**3))*np.sum(k_F_folded*weights)
    return n_ccs

## Maybe I can upgrade this, but it works well for now
##TODO: facing issues with convergence
def calculate_Fermi_velocity_along_direction(v_Fs, direction_vector, v_F_power_law=1):
    """
    From a list of the Fermi velocity vectors on a Fermi surface, interpolate the Fermi velocity magnitude in a specific direction.

    Assumes that there are not multiple Fermi velocities in the same direction.

    Parameters
    ----------
    v_Fs : ndarray([n,3])
        Fermi velocities of the mesh vertices
    direction_vector : ndarray([3])
        Direction vector to interpolate the Fermi velocity to
    v_F_power_law : float
        Power law for interpolation of the v_F magnitudes

    Returns
    -------
    float
        Fermi velocity magnitude in the specified direction
    """

    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    # Get the Fermi velocity directions and their magnitudes
    v_F_mag = np.linalg.norm(v_Fs, axis=1)
    v_F_mag_power = np.power(v_F_mag, v_F_power_law)
    v_F_normed = v_Fs / v_F_mag_power[:, np.newaxis]

    ## Only choose the Fermi velocities that are in the forward direction
    filter_forward_direction = np.dot(v_F_normed, direction_vector_normalized) > 0
    v_F_forward = v_Fs[filter_forward_direction]
    v_F_mag_forward = v_F_mag[filter_forward_direction]
    v_F_normed_forward = v_F_normed[filter_forward_direction]

    angles_from_direction = np.arccos(np.clip(np.dot(v_F_normed_forward, direction_vector_normalized), -1.0, 1.0))
    
    ## Fit only to elements near the desired direction
    element_filter = angles_from_direction <= np.percentile(angles_from_direction, 1)
    
    v_F_magnitudes_proj = np.dot(v_F_forward, direction_vector_normalized)
    v_F_proj_max = v_F_mag_forward[np.argmax(v_F_magnitudes_proj)]

    ## Function for the fit
    def parabolic_fit(x, A, B): return A*(1 + B*x**2)

    ## Initial values for the fit
    p0 = [v_F_proj_max, -1]

    # Use curve_fit to fit the parabolic function to the data
    try:
        popt, _ = curve_fit(parabolic_fit, angles_from_direction[element_filter], v_F_magnitudes_proj[element_filter], p0=p0, sigma=(angles_from_direction[element_filter]**2), maxfev=200000)
    except RuntimeError:
        print('Fit failed, returning max value')
        popt = [v_F_proj_max, 0]

    v_F_direction_interpolated, _ = popt

    v_F_direction_interpolated_exp = np.power(v_F_direction_interpolated, v_F_power_law)

    return v_F_direction_interpolated_exp

## Extremal orbits calculation

## This is slow and requires a fine mesh (or else it overestimates) but it does work
def calculate_cross_sectional_areas_from_slices(points : NDArray[np.float64], direction_vector : NDArray[np.float64], vector_plane_1 : NDArray[np.float64], num_slices : int = 51, take_center_slice : bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the cross-sectional areas of a Fermi surface from slices perpendicular to a given direction.

    Parameters
    ----------
    points : NDArray[np.float64] of shape (n,3)
        Points on the Fermi surface
    direction_vector : NDArray[np.float64] of shape (3)
        Direction vector perpendicular to desired cross-section
    vector_plane_1 : NDArray[np.float64] of shape (3)
        First vector defining the plane perpendicular to the direction vector
    num_slices : int, optional
        Number of slices to use for the calculation, by default 51
    take_center_slice : bool, optional
        Whether to return only the value of the center slice, by default False

    Returns
    -------
    cut_extremal_cross_sections : NDArray[np.float64] of shape (num_slices)
        Cross-sectional areas of the slices
    cut_centers : NDArray[np.float64] of shape (num_slices)
        Centers of the slices
    cut_boundaries : NDArray[np.float64] of shape (num_slices+1)
        Boundaries of the slices
    """
    assert np.dot(direction_vector, vector_plane_1) == 0, f'direction_vector {direction_vector} and vector_plane_1 {vector_plane_1} are not perpendicular'
   
    vector_plane_1 = vector_plane_1 / np.linalg.norm(vector_plane_1)

    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    ## Create the other vector of the plane
    vector_plane_2 = np.cross(vector_plane_1, direction_vector_normalized)

    k_F_projections = np.dot(points, direction_vector_normalized)

    extents_along_direction = (np.min(k_F_projections), np.max(k_F_projections))

    cut_boundaries = np.linspace(extents_along_direction[0], extents_along_direction[1], num_slices)
    cut_centers = (cut_boundaries[:-1] + cut_boundaries[1:])/2

    if take_center_slice:
        center_slices_index = np.argmin(np.abs(cut_centers))
        cut_centers = np.array([cut_centers[center_slices_index]], dtype=np.float64)
        cut_boundaries = np.array([cut_boundaries[center_slices_index], cut_boundaries[center_slices_index+1]])

    cut_extremal_cross_sections = np.zeros(len(cut_centers)) ## Initialize a vector of the correct length

    ## For each slice, calculate the area (2D volume) of the 
    for i in range(len(cut_extremal_cross_sections)):
        cut_boundary_1 = cut_boundaries[i]
        cut_boundary_2 = cut_boundaries[i+1]

        points_in_slice = points[(k_F_projections >= cut_boundary_1) & (k_F_projections <= cut_boundary_2)]

        points_2d = np.column_stack([
            np.dot(points_in_slice, vector_plane_1),
            np.dot(points_in_slice, vector_plane_2)
        ])
        
        hull = ConvexHull(points_2d)
        
        cut_extremal_cross_sections[i] = hull.volume

    return cut_extremal_cross_sections, cut_centers, cut_boundaries


def generate_cube_surface_points(side_length: float, num_points_side: int) -> NDArray[np.float64]:
    """
    Generate the points on the surface of a cube centered at the origin.

    Parameters
    ----------
    side_length: float
        Side length of the cube 
    num_points_side: int
        Number of points to use along each side of the cube

    Returns
    -------
    NDArray[np.float64]
        Points on the surface of the cube
    """

    half_side = side_length/2

     # Create arrays for each face
    face_points = []
    
    # Front and back faces (fixed z)
    x_face = np.linspace(-half_side, half_side, num_points_side)
    y_face = np.linspace(-half_side, half_side, num_points_side)
    X, Y = np.meshgrid(x_face, y_face)
    face_points.append(np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, -half_side))))
    face_points.append(np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, half_side))))
    
    # Left and right faces (fixed x)
    y_face = np.linspace(-half_side, half_side, num_points_side)
    z_face = np.linspace(-half_side, half_side, num_points_side)
    Y, Z = np.meshgrid(y_face, z_face)
    face_points.append(np.column_stack((np.full(Y.size, -half_side), Y.ravel(), Z.ravel())))
    face_points.append(np.column_stack((np.full(Y.size, half_side), Y.ravel(), Z.ravel())))
    
    # Top and bottom faces (fixed y)
    x_face = np.linspace(-half_side, half_side, num_points_side)
    z_face = np.linspace(-half_side, half_side, num_points_side)
    X, Z = np.meshgrid(x_face, z_face)
    face_points.append(np.column_stack((X.ravel(), np.full(X.size, -half_side), Z.ravel())))
    face_points.append(np.column_stack((X.ravel(), np.full(X.size, half_side), Z.ravel())))
    
    # Combine all faces
    k_F_cube_points = np.vstack(face_points)

    return k_F_cube_points

def calculate_cube_maximal_cross_section(side_length: float, direction: NDArray[np.float64], 
                                        vector_plane_1: NDArray[np.float64], num_points_side: int = 51) -> tuple[float, float]:
    """
    Calculate the maximal cross-sectional area perpendicular to a given direction
    for a cube with a given side length.

    Parameters
    ----------
    side_length: float
        Side length of the cube
    direction: NDArray[np.float64]
        Direction vector perpendicular to desired cross-section
    vector_plane_1: NDArray[np.float64]
        First vector defining the plane perpendicular to the direction vector
    num_points_side: int, optional
        Number of points to use along each side of the cube for numerical calculation, by default 51

    Returns
    -------
    float
        Maximal cross-sectional area
    float
        Center point of the maximal cross-sectional area. Linear along the direction vector.

    """
    ## Assert perpendularity
    assert np.dot(vector_plane_1, direction) == 0, 'vector_plane_1 and direction are not perpendicular'
    # Normalize the direction vector
    direction_normalized = direction / np.linalg.norm(direction)

    # Calculate the cross-sectional area
    # cross_section_area = side_length**2 * np.abs(np.dot(direction_normalized, np.array([1,1,1])))

    ## Generate points on the surface of a cube
    half_side = side_length/2

    cube_points = generate_cube_surface_points(side_length, num_points_side)

    cut_extremal_cross_sections, cut_centers, _ = calculate_cross_sectional_areas_from_slices(cube_points, direction_normalized, vector_plane_1)

    argmax_index = np.argmax(cut_extremal_cross_sections)
    cross_section_max_area = cut_extremal_cross_sections[argmax_index]
    cross_section_max_center = cut_centers[argmax_index]

    return cross_section_max_area, cross_section_max_center


def calculate_maximal_cross_section(points : NDArray[np.float64], direction : NDArray[np.float64], vector_plane_1 : NDArray[np.float64], num_points_side: int = 51):
    """
    Calculate the maximal cross-sectional area perpendicular to a given direction
    for a surface defined by a set of points.

    Parameters
    ----------
    points : NDArray[np.float64] of shape (n,3)
        Array of 3D coordinates defining the surface points
    direction : NDArray[np.float64] of shape (3)
        Direction vector perpendicular to desired cross-section
    vector_plane_1 : NDArray[np.float64] of shape (3)
        First vector defining the plane perpendicular to the direction vector
    num_points_side: int, optional
        Number of points to use along each side of the cube for numerical calculation, by default 51

    Returns
    -------
    float
        Maximal cross-sectional area
    """
    assert np.dot(direction, vector_plane_1) == 0, 'direction and vector_plane_1 are not perpendicular'
    
    # Normalize the direction vector
    direction_normalized = direction / np.linalg.norm(direction)
    
    cut_extremal_cross_sections, cut_centers, _ = calculate_cross_sectional_areas_from_slices(points, direction_normalized, vector_plane_1, num_points_side)

    argmax_index = np.argmax(cut_extremal_cross_sections)
    max_cross_section_area = cut_extremal_cross_sections[argmax_index]
    max_cross_section_center = cut_centers[argmax_index]

    return max_cross_section_area, max_cross_section_center
