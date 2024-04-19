This repository defines a methodology to allign raster data to aggregated stats of polygons. 
It comes from landcover/landuse applications, where often raster based remote sensing products predict certain landcover classes that don't match with official statistics. 
It can handle mixed aggregation levels as well, as long as the mixed levels are nested and a tree-structure is provided. 

This method takes in:
a) one raster layer for each landcover class x where each pixel contain the probability or conficence that that pixel is of class x. 
b) a geopandas vector object with the corresponding aggregated statistics to align to.
c) optional a fixed layer where the confidence is 100% and no alignment to admin data is performed.
d) as set of integers that should be used in the final output to indicate each class.


b) This geopandas needs to have one column for each landcover class that is to be aligned (called cl_{x}_totals) a unique name column NUTS_ID, a column called int_id that is a unique integer for each NUTS_ID and a column called tree. 
The tree column gives the relationship between the nested polygons. 
For example:
NorthAmerica
NorthAmerica/Mexico
NorthAmerica/USA
NorthAmerica/Canada
NorthAmerica/Canada/BC
NorthAmerica/Canada/ON
NorthAmerica/Canada/...
... For each territory


"NorthAmerica" and "BC" are then also in the column NUTS_ID. 