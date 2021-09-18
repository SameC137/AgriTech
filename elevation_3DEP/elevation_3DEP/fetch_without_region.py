import geopandas 

import pdal
import json
from shapely.geometry import shape, GeometryCollection, Polygon,Point,box
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import pyproj

from shapely.ops import transform
import pandas as pd


def getRegionsMeta():
    """Get Region meta dta frame

    Returns:
        [GeoDaraFrame]: Returns geodataframe containing metadata for all regions
    """
    pat=Path(__file__).parent.joinpath("metadata2.csv")
    frame=pd.read_csv(pat)
    frame["geometry"]= frame.apply(lambda x: box( x.xmin,x.ymin,x.xmax,x.ymax) ,axis=1)
    gdf=geopandas.GeoDataFrame(frame, geometry="geometry")
    gdf.set_crs(epsg=3857,inplace=True)
    return gdf

gdf=getRegionsMeta()
gdf.fillna(0,inplace=True)

def getRegion(boundary:geopandas.GeoDataFrame):
    """Get Region From Boundary

    Args:
        boundary (geopandas.GeoDataFrame): Boundary of region to be searched for

    Returns:
        [GeoDataFram]e: returns a geodataframe of the regions that contain the boundary
    """
    regions=gdf[gdf.geometry.contains(boundary.unary_union)]
    return regions




class FetchDataRegionLess:
    def __init__(self) -> None:
        pass
    def fetch_elevation(self,bound:geopandas.GeoDataFrame,crs:str)->dict:
        """Fetches elevation data for different years

        Args:
            bound (geopandas.GeoDataFrame): boundary dataframe
            crs (str): coordinate reference system of the output

        Returns:
            dict: dictionary of geodataframes as values and years as keys
        """
        regions=getRegion(bound.to_crs(epsg=3857))
        ret={}
        for index,i in regions.iterrows():
            data=self.fetch_elevation_region(bound,crs,i.filename)
            if i.year!=0:
                ret[i.year]=data
            else:
                ret["unknown"+str(index)]=data
        return ret
    def fetch_elevation_region(self,frame:geopandas.GeoDataFrame, crs:str,region:str):
        """Fetch Elevation for region

        Args:
            frame (geopandas.GeoDataFrame): The boundary for fetching the elevation
            crs (str): Coordinate reference system of the output
            region (str): The representative region string specifying file

        Returns:
            GeoDataFrame: GeoDataFrame containing the elevation data
        """
        try:
            pat=Path(__file__).parent.joinpath("fetch_template.json")
            with open(pat, 'r') as json_file:
                pipeline = json.load(json_file)
        except FileNotFoundError as e:
            print('FETCH_JSON_FILE_NOT_FOUND')
        
        
        boundary_repojected=frame.to_crs(epsg=3857)

        Xmin,Ymin,Xmax,Ymax=boundary_repojected.total_bounds
        pipeline[0]["filename"]=f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{region}ept.json"
        pipeline[0]["bounds"]=f"([{Xmin},{Xmax}],[{Ymin},{Ymax}])"
        pipeline[1]["polygon"]=boundary_repojected.geometry.unary_union.wkt
        pipeline[4]["out_srs"]="EPSG:"+ str(crs)
        # print( pipeline[0]["polygon"])
        
        pipe = pdal.Pipeline(json.dumps(pipeline))
        count = pipe.execute()
        arrays = pipe.arrays    
        metadata = pipe.metadata
        log = pipe.log
        years=[]
        for i in arrays:
            geometry_points = [Point(x, y) for x, y in zip(i["X"], i["Y"])]
            elevetions = i["Z"]
            frame=geopandas.GeoDataFrame(columns=["elevation", "geometry"])
            frame['elevation'] = elevetions
            frame['geometry'] = geometry_points
            frame.set_geometry("geometry",inplace=True)
            frame.set_crs(epsg=crs , inplace=True)
            years.append(frame)
        print("Done with "+region)
        return years[0]

   