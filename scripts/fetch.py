import geopandas as gpd

import pdal
import json
from shapely.geometry import shape, GeometryCollection, Polygon,Point,MultiPolygon
from typing import TypeVar, Union

import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from matplotlib.tri import Triangulation, LinearTriInterpolator,CubicTriInterpolator

import pyproj

from shapely.ops import transform
import numpy as np

class FetchData:
    def __init__(self,region:str) -> None:
        """Initialize Fetch With Region

        Args:
            region (str): region in list of regions found at https://s3-us-west-2.amazonaws.com/usgs-lidar-public/
        """
        self.region=region
    def fetch_elevation(self,boundary:gpd.GeoDataFrame, crs:str)->list[gpd.GeoDataFrame]:
        """Fetch Elevation Dataframe

        Args:
            boundary (GeoDataFrame): boundary defined in a geopandas dataframe
            crs (string): Coordinate reference system in string

        Returns:
            list[GeoDataFrame]: Returns a list of geopandas dataframes
        """
        try:
            pat=Path(__file__).parent.joinpath("fetch_template.json")
            with open(pat, 'r') as json_file:
                pipeline = json.load(json_file)
        except FileNotFoundError as e:
            print('FETCH_JSON_FILE_NOT_FOUND')
        
        boundary_repojected=boundary.to_crs(epsg=3857)

        Xmin,Ymin,Xmax,Ymax=boundary_repojected.total_bounds

        pipeline[0]["filename"]=f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{self.region}/ept.json"
        pipeline[0]["bounds"]=f"([{Xmin},{Xmax}],[{Ymin},{Ymax}])"
        pipeline[1]["polygon"]=boundary_repojected.geometry.unary_union.wkt
        pipeline[4]["out_srs"]=crs
        # print( pipeline[0]["polygon"])
        # print(pipeline)
        pipe = pdal.Pipeline(json.dumps(pipeline))
        count = pipe.execute()
        arrays = pipe.arrays    
        metadata = pipe.metadata
        log = pipe.log
        years=[]
        for i in arrays:
            geometry_points = [Point(x, y) for x, y in zip(i["X"], i["Y"])]
            elevetions = i["Z"]
            frame=gpd.GeoDataFrame(columns=["elevation", "geometry"])
            frame['elevation'] = elevetions
            frame['geometry'] = geometry_points
            frame.set_geometry("geometry",inplace=True)
            frame.set_crs(crs , inplace=True)
            years.append(frame)
        return years
    def repoject(self,polygon:Union[Polygon,Point,MultiPolygon],crs:str)->Union[Polygon,Point,MultiPolygon]:
        """Reproject Shape from crs to EPSG:3857

        Args:
            polygon (Union[Polygon,Point,MultiPolygon]): Polygon to be reporojected
            crs (str): Coordinate reference system

        Returns:
            Union[Polygon,Point,MultiPolygon]: Reprojected Polygon
        """
        
        wgs84 = pyproj.CRS(crs)
        utm = pyproj.CRS('EPSG:3857')

        project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        utm_point = transform(project, polygon)
        return utm_point
    def visualize3D(self,data:gpd.GeoDataFrame)->None:
        """Create a 3D pointcloud Visualization

        Args:
            data (GeoDataFrame): Dataframe to be visualized
        """ 
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter( data.geometry.x,data.geometry.y,data.elevation, cmap='Spectral_r',s=0.001, c=data.elevation)
        plt.show()
    def visualize2D(self,data:gpd.GeoDataFrame)->None:
        """Create a 2D visualization

        Args:
            data (GeoDataFrame): Dataframe to be visualized
        """
        fig, ax = plt.subplots(1, 1)
        data.plot(column='elevation',  cmap = 'Spectral_r', ax=ax, legend=True)
        plt.show()
    def standardize(self,data,resolution):
        """Standardize and interpolate dataframe

        Args:
            data (GeoDataFrame): Dataframe to be standardized
            resolution (int): Resolution of the standardization

        Returns:
            GeoDataFrame: Returns the interpolated and standardized dataframe
        """
                
        data_meters= data.to_crs({'init': 'epsg:3395'})
        
        rasterRes = resolution
        
        totalPointsArray = np.zeros([data_meters.shape[0],3])
        for index, point in data_meters.iterrows():
            pointArray = np.array([point.geometry.coords.xy[0][0],point.geometry.coords.xy[1][0],point['elevation']])
            totalPointsArray[index] = pointArray
        xCoords = np.arange(totalPointsArray[:,0].min(), totalPointsArray[:,0].max()+rasterRes, rasterRes)
        yCoords = np.arange(totalPointsArray[:,1].min(), totalPointsArray[:,1].max()+rasterRes, rasterRes)
        zCoords = np.zeros([yCoords.shape[0],xCoords.shape[0]])
        polygons=[]
        elevations=[]
        
        triFn = Triangulation(totalPointsArray[:,0],totalPointsArray[:,1])
        #linear triangule interpolator funtion
        linTriFn = LinearTriInterpolator(triFn,totalPointsArray[:,2])
        #loop among each cell in the raster extension
        for indexX, x in np.ndenumerate(xCoords):
            for indexY, y in np.ndenumerate(yCoords):
                tempZ = linTriFn(x,y)
                #filtering masked values
                if tempZ == tempZ:
                    zCoords[indexY,indexX]=tempZ
                    
                    polygons.append(Point(x,y)) 
                    elevations.append(float(tempZ.data[()]))
                else:
                    zCoords[indexY,indexX]=np.nan
        frame=gpd.GeoDataFrame(columns=["elevation", "geometry"])
        frame['elevation'] = elevations
        frame['geometry'] = polygons
        frame.set_geometry("geometry",inplace=True)
        frame.set_crs(crs=data_meters.crs, inplace=True)
        return frame.to_crs(data.crs)