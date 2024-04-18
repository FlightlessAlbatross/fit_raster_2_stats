#!/usr/bin/env python
# coding: utf-8



import numpy as np
import rasterio
from rasterio.mask import mask
import os

from rasterio.features import bounds as feature_bounds
from rasterio.transform import from_origin
import pandas as pd

import geopandas as gpd

import fiona

import numpy.ma as ma # for masked arrays. This allows us to handle NAs for integer arrays


# for the blur
from scipy.ndimage import convolve, median_filter

# for the custom class
from dataclasses import dataclass, field
from typing import List

import heapq

import time



# In[9]:


# DONE TODO: Introduce a map between regional (here NUTS) IDs and an integer. --> we do this by having a final leaf after the NUTS3 area that is the interger code. 
# then use the integer internally. This frees us from having to use NUTS specifically and opens us up to use non-numeric NUTS codes.

# TODO: create tree structure and counter from data. 
# TODO: create rasterized nuts3/2 for all of europe with integers and create tree structure from that. 


# TODO: nice output for the treecounter. 
# # summarize on the smallest common level
# # print out full structure


# In[15]:


class TreeStructure:
    def __init__(self, paths):
        self.structure = self._build_tree_from_paths(paths)
#         self._integrety_check()
        self.end_node_to_full_path = self._generate_end_node_mappings(paths)
        
       

    def _build_tree_from_paths(self, paths):
        tree = {}
        for path in paths:
            parts = path.split('/')
            current = tree
            for part in parts:
                current = current.setdefault(part, {})  # Use setdefault for cleaner code
        return tree

    def _generate_end_node_mappings(self, paths):
        return {path.split('/')[-1]: path for path in paths}
    
    def _integrety_check(self):
        # check that every subnode is also a node. 
        # if A/B/C is a node, then both A and A/B need to be nodes.
        
        pass

    def get_full_path(self, end_node):
        return self.end_node_to_full_path.get(end_node)
    
    def get_paths_from_leaf(self, end_node):
        path = self.get_full_path(end_node)
        all_paths = [path]
        for p in reversed(path.split('/')[:-1]):
            all_paths.append(self.get_full_path(p))

        return all_paths
    
    def get_child_nodes(self, node_identifier):
        current = self.structure
        for part in self.get_full_path(node_identifier).split('/') if node_identifier else []:
            current = current.get(part, {})
        return list(current.keys()) if current else []

    def get_youngest_children_nodes(self, node_identifier):
        def collect_leaves(node):
            return [key for key, subtree in node.items() if not subtree] if node else []
        current = self.structure
        for part in self.get_full_path(node_identifier).split('/') if node_identifier else []:
            current = current.get(part, {})
        return collect_leaves(current)
    
    def get_all_leaf_nodes(self):
        """Return a list of all leaf nodes (nodes with no children) in the entire tree."""
        def collect_all_leaves(node):
            leaves = []
            for key, subtree in node.items():
                if not subtree:  # No children means this is a leaf node
                    leaves.append(key)
                else:
                    leaves.extend(collect_all_leaves(subtree))
            return leaves
        
        return collect_all_leaves(self.structure)
    
    
    def full_subtree(self, nodes):
        '''Return the full tree from the root to the given endnodes'''
        full_branches_of_nodes = set()
        for node in nodes:
            '''Split them into separate branches'''
            full_branch = self.get_paths_from_leaf(node)
            
            if len(full_branch) == 0:
                print(f'The full_subtree subroutine found an empty branch.')
                raise TypeError
            
            if len(full_branch) == 1:
                full_branches_of_nodes.add(full_branch[0])
            else:
                full_branches_of_nodes.update(full_branch)
                
                
        return full_branches_of_nodes
    
    def smallest_subtree(self, nodes):
        '''Return the smallest subtree with a single root that contains the given nods as endnodes'''
        
        if not isinstance(nodes, list):
            nodes = [nodes]
        
        tree_paths = self.full_subtree(nodes)   
        path_parts = [path.split('/') for path in tree_paths]
        path_parts.sort(key=len, reverse = False)  # Sort paths by their depth (length)


        # Identify the largest possible root that is common to all paths
        smallest_root = None
        for i in range(len(path_parts)):  # Iterate over parts of the shortest path
            part = path_parts[i]  # Candidate for the smallest root
            if all(part == p[:i + 1] for p in path_parts[i:]):
                longest_root = part
            else:
                break


        longest_root

        if longest_root is None:
            return set()  # No common root found
        

        # Rebuild the smallest_root into a path string
        longest_root_path = '/'.join(longest_root)

        # Filter the original set to include only descendants of the smallest root
        filtered_tree = {path for path in tree_paths if path.startswith(longest_root_path)}

        return filtered_tree
        

class TreeCounters:
    def __init__(self, tree_structure):
        self.tree_structure = tree_structure
        self.counters = {}

    def add_counter(self, key, counter_path, initial_value):
        self.counters.setdefault(key, {})[counter_path] = initial_value

    def node_has_counter(self, key, end_node):
        full_path = self.tree_structure.get_full_path(end_node)
        return full_path in self.counters.get(key, {}) if full_path else False

    def top_node_needed_for_counter(self, end_node):
        path = self.tree_structure.get_full_path(end_node)
        layers = self.counters.keys()
        highest_node, max_traversal = end_node, 0
        if path:
            parts = path.split('/')
            for layer in layers:
                for i, part in enumerate(reversed(parts)):
                    if self.node_has_counter(layer, part):
                        if i > max_traversal:
                            max_traversal, highest_node = i, part
                        break
        return highest_node
          
    
    def split_tree_2_endnodes(self):
        '''Split the end_nodes of the tree (with counters) into the smallest groups with valid counters'''
        seen_regions = set()
        leafs_of_smallest_subtrees = []
        for region in self.tree_structure.get_all_leaf_nodes():

            if region in seen_regions:
                continue

            leafs_of_subtree = (self.verify_and_extend_subtree(region))
            seen_regions.update(leafs_of_subtree)
            leafs_of_smallest_subtrees.append(leafs_of_subtree)
            
            
        return(leafs_of_smallest_subtrees)
    
    def split_tree(self):
        '''Split the end_nodes of the tree (with counters) into the smallest groups with valid counters'''
        '''Return the full tree structure of the smallest possible tree that contains the split endnodes'''
        all_complete_subtrees_endnodes = self.split_tree_2_endnodes()
        all_subtrees = []
        for endnodes in all_complete_subtrees_endnodes:
            subtree = self.tree_structure.smallest_subtree(endnodes)
            all_subtrees.append(subtree)
            
            
        '''This is just for statistics, not needed for the final output'''
        subtree_sizes = [len(tree) for tree in all_subtrees]    

        size_1_counter = 0
        max_size = 0
        for treesize in subtree_sizes:
            if treesize == 1:
                size_1_counter += 1
            if treesize > max_size:
                max_size = treesize

        print(f'We split the whole tree into {len(all_subtrees)} subtrees.')
        print(f'{size_1_counter} of these subtrees are a single region.')
        print(f'The largest subtree has {max_size} (nested) regions')

        '''End of statistics'''
        
        return(all_subtrees)



    def verify_and_extend_subtree(self, start_node):
        """Verifies and possibly extends the subtree starting from 'start_node' to ensure all youngest nodes have required counters."""
        youngest_children = self.tree_structure.get_youngest_children_nodes(start_node)
        if not youngest_children:  # If no children, consider the start_node as a leaf
            youngest_children = [start_node]

        extended_children = set()
        all_children_satisfied = True
        layers = self.counters.keys()

        for child in youngest_children:
            satisfied = True
            for layer in layers:
                if not self.node_has_counter(layer, child):
                    satisfied = False
                    break
            if satisfied:
                extended_children.add(child)
            else:
                all_children_satisfied = False
                top_node = self.top_node_needed_for_counter(child)
                if top_node != child:  # Avoid recursion on the same node
                    extended_children.update(self.tree_structure.get_youngest_children_nodes(top_node))
                else:
                    extended_children.add(child)

        if all_children_satisfied or not extended_children.difference(youngest_children):
            return list(youngest_children)
        else:
            # Move up to the top node necessary for missing counters
            highest_needed_node = start_node
            for layer in layers:
                node_needed = self.top_node_needed_for_counter(start_node)
                if self.tree_structure.get_paths_from_leaf(node_needed)[-1] in self.tree_structure.get_paths_from_leaf(highest_needed_node):
                    highest_needed_node = node_needed
            if highest_needed_node != start_node:
                return self.verify_and_extend_subtree(highest_needed_node)
            else:
                return list(extended_children)

    def current_counter(self, key, end_node):
        """Find the nearest counter up the tree from the end node and decrement it."""
        if key not in self.counters:
            # print(f"No counters found for this class {key}.")
            return
        
        
        full_path = self.tree_structure.get_full_path(end_node)
        if not full_path:
            print(f"No full path found for end node {end_node}.")
            return

        for path in self.tree_structure.get_paths_from_leaf(end_node):
            if path in self.counters[key]:
                return self.counters[key][path] 


    def decrement(self, key, end_node, amount=1):
        full_path = self.tree_structure.get_full_path(end_node)
        if full_path:
            for path in self.tree_structure.get_paths_from_leaf(end_node):
                if path in self.counters.get(key, {}):
                    self.counters[key][path] -= amount
                    return
            print(f"No counter found for {end_node} under key {key}.")
        else:
            print(f"No full path found for end node {end_node}.")


@dataclass
class LayerConfig:
    paths: List[str]
    codes: List[int]

    def __post_init__(self):
        self.validate_layers()

    def validate_layers(self):
        if len(self.paths) != len(self.codes):
            raise ValueError("Paths and codes lists must have the same length.")
        self.check_raster_properties()

    def check_raster_properties(self):
        properties = None
        for path in self.paths:
            with rasterio.open(path) as src:
                if properties is None:
                    properties = (src.width, src.height, src.transform)
                elif (src.width, src.height, src.transform) != properties:
                    raise ValueError("All input paths must have the same raster size, resolution, and origin.")
                    
    def get_gis_info(self):
        with rasterio.open(self.paths[0]) as src:
            west, north = src.bounds.left, src.bounds.top
            res_x, res_y = src.res
            epsg_code = src.crs.to_epsg()
            return west, north, res_x, res_y, epsg_code
                    
@dataclass
class ProcessingConfig:
    fixed_layer: LayerConfig
    probability_layer: LayerConfig
    administrative_polygon: gpd.GeoDataFrame
    tree_counters: TreeCounters = field(init=False)


    def __post_init__(self):
        self._validate_config()
        self._initialize_tree_counters()

    def _validate_config(self):
        # Check for duplicate codes between fixed and probability layers
        fixed_codes = set(self.fixed_layer.codes)
        probability_codes = set(self.probability_layer.codes)
        if fixed_codes.intersection(probability_codes):
            print("Warning: There are duplicate codes between fixed and probability layers.")

        # Validate administrative_polygon
        self._validate_polys()

    def _validate_polys(self):
        
        number_features = self.administrative_polygon.shape[0]
        if number_features == 0:
            raise ValueError("No Polygons given")
        
#         if number_features > 1:
#             raise ValueError("Right now only a single polygon is allowed in the administrative polygon")
        
        required_columns = ['NUTS_ID', 'tree']
        for code in self.probability_layer.codes:
            required_columns.append(f"cl_{code}_total")  # check if every probability layer has a corresponding total in the admin data
            

        missing_columns = [col for col in required_columns if col not in self.administrative_polygon.columns]
        if missing_columns:
            raise ValueError(f"administrative_polygon is missing required columns: {missing_columns}")
            
        
        # check that all data columns are integers.
        for col in required_columns[1:]:
            if isinstance(self.administrative_polygon[col], int):
                continue
            else:
                pass
# (f"TODO: Check that data columns are integers")                


    def _initialize_tree_counters(self):
        tree_structure = TreeStructure(self.administrative_polygon['tree'].tolist())
        self.tree_counters = TreeCounters(tree_structure)
        self._populate_treecounter()

    def _populate_treecounter(self):
        for i, row in self.administrative_polygon.iterrows():
            treepath = row['tree']
            for code in self.probability_layer.codes:
                datacol = f'cl_{code}_total'
                value = row[datacol]
                if value != -99:  # Assuming -99 is used to indicate missing or irrelevant data
                    self.tree_counters.add_counter(code, treepath, value)

                
        
        # Further checks can be added here, e.g., checking that the polygon extent is within the raster layers' extents.

    def polygon_ids(self):
        return self.administrative_polygon.NUTS_ID.tolist()
    
    
    def make_dict_integer_2_region(self):
        return {code : integer for (code, integer) in zip(self.administrative_polygon['int_id'], self.administrative_polygon['NUTS_ID'])}
    
 
    
        
        
    def get_poly_GIS_info_for_writing(self, geo_id = None):
        # resolution and EPSG come from a raster
        
        raster_path = self.probability_layer.paths[0]
        
        with rasterio.open(raster_path) as src:
                
                if geo_id is None:
                    if len(self.administrative_polygon.geometry) != 1:
                        print('Warning, no geo_id was passed to get_poly_GIS. First one was taken by default')
                    geometry = self.administrative_polygon.geometry.iloc[0]
                else:
                    index = self.administrative_polygon.index[self.administrative_polygon['NUTS_ID'] == geo_id][0]
                    geometry = self.administrative_polygon.at[index, 'geometry']

                out_image, out_transform = mask(src, [geometry], invert=False, crop=True)
                west =  out_transform[2]
                north = out_transform[5]
                resolution = src.res  # (pixel width, pixel height)

                # Get CRS
                crs = src.crs
                if crs is not None:
                    # Extract EPSG code
                    epsg_code = crs.to_epsg()  # This will be None if the EPSG code can't be determined 
                return (west, north, resolution[0], resolution[1], epsg_code)

                
    
    
                


def get_mask_from_polygon(input_path, poly, na_value=255):
    with rasterio.open(input_path) as src:
        
        
        transform = src.transform
        width = src.width
        height = src.height

        # Calculate the coordinates of the corners
        # top-left corner
        top_left_x, top_left_y = transform * (0, 0)
        # bottom-right corner
        bottom_right_x, bottom_right_y = transform * (width, height)

        # Get the bounding box (extent)
        extent = [min(top_left_x, bottom_right_x),
                  min(top_left_y, bottom_right_y),
                  max(top_left_x, bottom_right_x),
                  max(top_left_y, bottom_right_y)]       
        
        geometry = poly.geometry.iloc[0]
        out_image, out_transform = mask(src, [geometry], invert=False, crop=True)
        
        # Set values not equal to na_value to 0 (or another nodata value of your choice)
        out_image[out_image != na_value] = 0
        
        # The result is a 3D array: (bands, rows, cols). If you have a single band, you can squeeze it to 2D
        if out_image.shape[0] == 1:
            return out_image.squeeze()  # Removes the bands dimension if it's 1
        else:
            return out_image  # Returns the full array if there are multiple bands


# In[5]:


def read_raster(input_path, poly):
    with rasterio.open(input_path) as src:
            geometry = poly.geometry.iloc[0]
            nutsid = poly['NUTS_ID']

            # Mask and crop the raster using the current polygon
            out_image, out_transform = mask(src, [geometry], invert=False, crop=True)

            out_image = out_image.astype(float)  # Ensure the data is in float
            out_image[out_image == 255] = np.nan


            resolution = src.res  # (pixel width, pixel height)

            # Get CRS
            crs = src.crs
            if crs is not None:
                # Extract EPSG code
                epsg_code = crs.to_epsg()  # This will be None if the EPSG code can't be determined


            affine = out_transform


            if len(out_image.shape) == 3:
                array = np.squeeze(out_image, axis = 0)

            # mask the 255 values
            array_mask = np.isnan(array)

            array = ma.array(array, mask = array_mask)

            return array


# In[ ]:


def make_region_raster(polys, reference_raster):
    # it might be more convinient to create the rasterized IDs on the fly instead of creating it first and passing it. 
    # this function is not done though. 
        
    geometry = polys.geometry.tolist()
    
    with rasterio.open(reference_raster) as src:
        out_image, out_transform = mask(src, geometry, invert=False, crop=True)
        
     


 # make a stack of smeared probability layers
def apply_read_raster_2_probability_stack(probability_layer_paths, poly):
    temp = [read_raster(layer, poly) for layer in probability_layer_paths]
    return np.ma.stack(temp)


# In[22]:


def Class_label_dictionaries(codes, reverse = False):
    
    if reverse:
        return {classlabel:position for (position, classlabel) in enumerate(codes)}

    else:
        return {position:classlabel for (position, classlabel) in enumerate(codes)}

    

def flatten_layers_withregion(probability_layer, probability_layer_codes, region_raster, int_2_region):
    
    # get the dictionary from array index to class name (int)
    layer_idx_2_final = Class_label_dictionaries(probability_layer_codes, reverse = False)
    

    # Create a combined mask for NaN, zeros, and existing masked values
    # fill in the mask values otherwise we can't combine htem. 
    # it doesn't matter what we choose T/F as by definition the existing mask is already True there and we or them later. 
    
    nan_mask = np.isnan(probability_layer).filled(True)   # Mask where NaN values are True
    zero_mask = (probability_layer == 0).filled(True)  # Mask where zero values are True
    existing_mask = ma.getmaskarray(probability_layer)  # Get the existing mask from the masked array

    # Combine the masks
    combined_mask = nan_mask  | existing_mask | zero_mask
    
    # take the probability layer stack and extract the pixel values, together with the x,y,z (layer) location. 
    # also translate the layer postion (i) to the final class code. 
    
    # one tuple is of the form 
    # (row, column, probability, region)

    flattened_with_indices = {}
    for i in range(probability_layer.shape[0]): # layers 
                flattened_with_indices[layer_idx_2_final[i]] = [(j, k, probability_layer[i, j, k], int_2_region [int(region_raster[j,k])]  ) 
                              for j in range(probability_layer.shape[1]) # rows
                              for k in range(probability_layer.shape[2]) # cols
                              if not combined_mask[i, j, k] 
                             ]

    return flattened_with_indices


# In[24]:


def order_flat_layers(flattened_with_indices):
    for layer in flattened_with_indices.values():
        # Sort each list in place by the 4th element of each tuple, in descending order.
        layer.sort(key=lambda x: x[2], reverse=True)



# In[26]:


def remove_highest(dict_of_tuples):
    # Invert the lists to easily pop the last tuple (assuming it's the "largest" based on the third element)
    for key in dict_of_tuples.copy():
        dict_of_tuples[key] = list(reversed(dict_of_tuples[key]))
    
    # Initialize heap with the "largest" tuple from each list based on the third element
    heap = []
    for key, lst in dict_of_tuples.items():
        if lst:  # Check if the list is not empty
            # Push a tuple containing the negated third element for max heap simulation, the key, and the tuple
            heapq.heappush(heap, (-lst[-1][2], key, lst.pop()))
    
    while heap:
        _, key, tuple_value = heapq.heappop(heap)  # Pop the tuple with the "largest" third element
        yield (key, tuple_value ) # Yield the original tuple
        
        if dict_of_tuples[key]:  # If there are more tuples in the list
            # Push the next "largest" tuple from the same list
            heapq.heappush(heap, (-dict_of_tuples[key][-1][2], key, dict_of_tuples[key].pop()))


# In[27]:


def initialize_output_with_fixed_layers(fixed_layer_paths, fixed_layer_codes, array_mask, poly, misc_value):
    
    geometry = poly.geometry.iloc[0]
    
    output = np.full(array_mask.shape, misc_value, dtype = 'int')
    # mask the NAs. 
    output = ma.array(output, mask = array_mask )
    
    for fixed_layer_path, class_final_code in zip(fixed_layer_paths, fixed_layer_codes):
        with rasterio.open(fixed_layer_path) as src:
            fixed_mask, fixed_mask_transform = mask(src, [geometry], invert=False, crop=True)
    
        for i in range(fixed_mask.shape[1]):
            for j in range(fixed_mask.shape[2]):
                if fixed_mask[0,i,j] == 1:
                    output[i,j] = class_final_code
            
    return output
        


        





def fill_output_array(initial_output, probability_layers, treecounter, misc_value):

    
    # array with the correct shape, and the fixed layers set. 
    # all other pixels are masked or the misc value. 
    output = initial_output

    # remove_highest is a generator and returns the highest value between all layers with popping them off - repeatetly.
    for layer, tupl in remove_highest(probability_layers): 
        
        # layer is the layer encoded as its final value (e.g. 1 = forest)
        # tupl is a tupl with 4 elements 0:3
        # the first 2 are row/col of the raster
        # the 3rd is the probability
        # the 4th is the integer value encoding the polygon region
        
        idx = tupl[0:2]

        if output[idx] != misc_value:
            # this means the pixel has been claimed by a fixed, or probability layer. 
            # NA pixels should not appear in the flattend_sorted object.
            continue
            
        current_counter = treecounter.current_counter(layer, tupl[3])
        
        if current_counter is not None: # check if there is a counter for that
            
            # if current_counter % 100000 == 1:
            #     print(f"The current counter for {tupl[0]} in {tupl[4]} is {current_counter}")
            
        # if the class is in our counter, check if the counter is 0
        # # if it is, go to the next entry
        # # if it is not, assign the pixel, add it to the seen pixels and decriment the counter
        # if it is not in the counter assign it and add it to the seen pixels. We don't have a restriction on this class. 
                      
            if current_counter < 1: # is the counter 0?
                continue

            else: # if it isn't use the current label and decrement the corresponding counter.

                output[idx] = layer
                treecounter.decrement(layer, tupl[3])       
        
        else:
            output[idx] = layer           
            
#     print(f"The final counter is {counter}. It should be all 0s. A formal check can be added TODO.")
    return output





def pipeline(config: ProcessingConfig, region_raster_path, misc_value = 10, output_folder = None):

    fixed_layer_paths = config.fixed_layer.paths
    fixed_layer_codes = config.fixed_layer.codes
    probability_layer_paths = config.probability_layer.paths
    probability_layer_codes = config.probability_layer.codes
    administrative_polygon = config.administrative_polygon
        

    all_complete_subtrees = config.tree_counters.split_tree()
    
    if output_folder is None: #create an output container if we don't write to disc. 
        print('The output will be returned as a dictionary of arrays corresponding to the subtrees. It is recommended to supply an output folder such that each intermediary step gets written to disc. ')
        output_dict = {}

    for subtree in all_complete_subtrees:
        
        smallest_complete_poly_branch = min(subtree, key=len)
        top_node = smallest_complete_poly_branch.split('/')[-1]
        
#         print(f"top node of the smallest complete Poly: {smallest_complete_poly_branch}")
        
        smallest_complete_poly = administrative_polygon.loc[administrative_polygon['tree'] == smallest_complete_poly_branch]
    
        all_polys = administrative_polygon['tree'].isin(subtree)
        
        # TODO!!!! maybe create this on the fly, rather than having one on disc? its one less thing to add
        region_raster = read_raster(region_raster_path, smallest_complete_poly)

        
        array_mask = get_mask_from_polygon(probability_layer_paths[0], smallest_complete_poly)

        probability_layer = apply_read_raster_2_probability_stack(probability_layer_paths, 
                                                                             smallest_complete_poly)
        
        int_2_regions = config.make_dict_integer_2_region()

        flattened_with_indices = flatten_layers_withregion(probability_layer, probability_layer_codes, region_raster, int_2_regions)


        order_flat_layers(flattened_with_indices)

        initial_output = initialize_output_with_fixed_layers(fixed_layer_paths, fixed_layer_codes,
                                                            array_mask, smallest_complete_poly, 10)


        pipe = fill_output_array(initial_output, flattened_with_indices, config.tree_counters, 10)
        
        
        if output_folder:
            output_path = f'{output_folder}/{top_node}_admin_prediction.tif'
            write_final_output(config, pipe, output_path, top_node)
        else:
            output_dict[top_node] = pipe
            
        print(f'fill_output_array for subtree below {top_node} Done')
        
    if output_folder is none:
        return output_dict
    else:
        print(f"All subtree rasterlayers have been writen to {output_folder}. Return 0")
        return 0
        

def write_final_output(config: ProcessingConfig, data, output_path, top_node):

    west, north, res_x, res_y, epsg_code = config.get_poly_GIS_info_for_writing(top_node)
    

    # Define the transformation and metadata
    transform = from_origin(west, north, res_x, res_y)
    height, width = data.shape
    crs = f"EPSG:{epsg_code}"  # Example CRS - replace with the appropriate CRS for your data
    
    # Metadata dictionary
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'int16',
        'crs': crs,
        'transform': transform,
        'nodata': 255 
    }


    # Write to a new TIFF file
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(data, 1)


