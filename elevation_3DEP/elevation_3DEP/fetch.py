

import pdal
import json
import geopandas as gpd
from shapely.geometry import shape, GeometryCollection, Polygon,Point,MultiPolygon

from shapely.ops import transform
from typing import TypeVar, Union

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from scipy.interpolate import griddata
from matplotlib.tri import Triangulation, LinearTriInterpolator,CubicTriInterpolator
from math import floor
import math
import pyproj

import numpy as np

from elevation_3DEP.logger_util import CreateLogger 

logger = CreateLogger('FetchData')
logger = logger.get_default_logger()

class FetchData:
    def __init__(self,region:str) -> None:
        """Initialize Fetch With Region

        Args:
            region (str): region in list of regions found at https://s3-us-west-2.amazonaws.com/usgs-lidar-public/
        """
        self.region=region
        # logger.info('Initialized fetch object')
    def fetch_elevation(self,boundary:gpd.GeoDataFrame, crs:str)->list[gpd.GeoDataFrame]:
        """Fetch Elevation Dataframe

        Args:
            boundary (GeoDataFrame): boundary defined in a geopandas dataframe
            crs (string): Coordinate reference system in string

        Returns:
            list[GeoDataFrame]: Returns a list of geopandas dataframes
        """
        
        # logger.info('Fetch point cloud data started')
        try:
            pat=Path(__file__).parent.joinpath("fetch_template.json")
            with open(pat, 'r') as json_file:
                pipeline = json.load(json_file)
        except FileNotFoundError as e:
            logger.exception('Fetch JSON template File not found')
        
        boundary_repojected=boundary.to_crs(epsg=3857)

        Xmin,Ymin,Xmax,Ymax=boundary_repojected.total_bounds

        pipeline[0]["filename"]=f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{self.region}/ept.json"
        pipeline[0]["bounds"]=f"([{Xmin},{Xmax}],[{Ymin},{Ymax}])"
        pipeline[1]["polygon"]=boundary_repojected.geometry.unary_union.wkt
        pipeline[4]["out_srs"]=crs
        # print( pipeline[0]["polygon"])
        # print(pipeline)
        try:
            pipe = pdal.Pipeline(json.dumps(pipeline))
            count = pipe.execute()
            arrays = pipe.arrays    
            metadata = pipe.metadata
        except:
            logger.info('Error occured: Could not fetch point cloud data')

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
        
        
        logger.info('Fetch point cloud data completed')

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
        
        # logger.info('Visualization 3D Started')
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter( data.geometry.x,data.geometry.y,data.elevation, cmap='Spectral_r',s=0.001, c=data.elevation)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation')
        plt.show()
        return fig
        # logger.info('Visualization 3D complete')
    def visualize2D(self,data:gpd.GeoDataFrame)->None:
        """Create a 2D visualization

        Args:
            data (GeoDataFrame): Dataframe to be visualized
        """
        
        logger.info('Visualization 2D complete')
        fig, ax = plt.subplots(1, 1)
        try:
            data.plot(column='elevation',  cmap = 'Spectral_r', ax=ax, legend=True)
        except KeyError:
            logger.exception('Elevation column does not exist')
        except:
            logger.exception("Exception Occured: Could not complete visualization")

        plt.show()
        
        logger.info('Visualization 2D complete')
    def standardize(self,data,resolution):
        """Standardize and interpolate dataframe

        Args:
            data (GeoDataFrame): Dataframe to be standardized
            resolution (int): Resolution of the standardization

        Returns:
            GeoDataFrame: Returns the interpolated and standardized dataframe
        """
        
        logger.info('Standardization Started')

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
        
        logger.info('Grid Created Proceeding to interpolation')
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
        logger.info("Interploation Completed")
        frame=gpd.GeoDataFrame(columns=["elevation", "geometry"])
        frame['elevation'] = elevations
        frame['geometry'] = polygons
        frame.set_geometry("geometry",inplace=True)
        frame.set_crs(crs=data_meters.crs, inplace=True)
        
        logger.info('Standardization Completed')
        return frame.to_crs(data.crs)
    def standardize2(self,data,resolution):
        
        logger.info('Standardization Started')
        data_meters= data.to_crs({'init': 'epsg:3395'})
        
        rasterRes = resolution
        try:
            points = [[i.x,i.y] for i in data_meters.geometry.values]
        except KeyError:
            logger.exception('Geometry column does not exist')
        except:
            logger.exception("Exception Occured: Could not complete Standardization")   

        try:
            values = data_meters['elevation'].values
        except KeyError:
            logger.exception('Elevation column does not exist')
        except:
            logger.exception("Exception Occured: Could not complete visualization")    

        xDim = floor(data_meters.total_bounds[2])-floor(data_meters.total_bounds[0])
        yDim = floor(data_meters.total_bounds[3])-floor(data_meters.total_bounds[1])
        # rasterRes = 0.5
        nCols = xDim / rasterRes
        nRows = yDim / rasterRes

        
        grid_y, grid_x = np.mgrid[floor(data_meters.total_bounds[1])+rasterRes/2:floor(data_meters.total_bounds[3])-rasterRes/2:nRows*1j,
                                floor(data_meters.total_bounds[0])+rasterRes/2:floor(data_meters.total_bounds[2])+rasterRes/2:nCols*1j]
        
        logger.info('Grid Created Proceeding to interpolation')
        mtop = griddata(points, values, (grid_x, grid_y), method='linear')
        
        logger.info("Interploation Completed")
        x=grid_x.flatten()
        y=grid_y.flatten()
        geometry_points = [Point(x, y) for x, y in zip(x, y)]
        elevetions = mtop.flatten()
        frame=gpd.GeoDataFrame(columns=["elevation", "geometry"])
        frame['elevation'] = elevetions
        frame['geometry'] = geometry_points
        frame.set_geometry("geometry",inplace=True)
        frame.set_crs("epsg:3395" , inplace=True)
        
        logger.info("Standardization Completed")
        return frame.to_crs(data.crs)
    def topographicWetnessIndex(self,data, resolution):
        """Calculate Topographic Wetness Index for elevations

        Args:
            data (GeoDataFRame): GeoDataFrame containing points and thier elevation
            resolution (int): Resolution of the generated TWI 

        Returns:
            GeoDataFrame: GeoDataFrame With a new Column For the index
        """
        
        logger.info("Topographic Wetness Index Generation Started")
        data_meters= data.to_crs({'init': 'epsg:3395'})
        
        rasterRes = resolution
        try:
            points = [[i.x,i.y] for i in data_meters.geometry.values]
        except KeyError:
            logger.exception('Geometry column does not exist')
        except:
            logger.exception("Exception Occured: Could not complete Standardization")   
            
        try:
            values = data_meters['elevation'].values
        except KeyError:
            logger.exception('Elevation column does not exist')
        except:
            logger.exception("Exception Occured: Could not complete visualization")    


        xDim = round(data_meters.total_bounds[2])-round(data_meters.total_bounds[0])
        yDim = round(data_meters.total_bounds[3])-round(data_meters.total_bounds[1])
        # rasterRes = 0.5
        nCols = xDim // rasterRes
        nRows = yDim // rasterRes

        grid_y, grid_x = np.mgrid[round(data_meters.total_bounds[1])+rasterRes/2:round(data_meters.total_bounds[3])-rasterRes/2:nRows*1j,
                                round(data_meters.total_bounds[0])+rasterRes/2:round(data_meters.total_bounds[2])+rasterRes/2:nCols*1j]
        mtop = griddata(points, values, (grid_x, grid_y), method='linear')

        slopes=[]

        slopeMatrix = np.zeros([nRows,nCols])
        for indexX in range(0,nCols):
            for indexY in range(0,nRows):
                # print(indexX,indexY)
                # if not np.isnan(zCoords[indexY,indexX]):
                # slopeMatrix
                
                mat=neighbors(mtop,indexY+1,indexX+1)
                ind=twi(mat,rasterRes)
                # if indexX[0]>508:
                #     print(pointSlope,indexY)
                slopeMatrix[indexY,indexX]=ind
                slopes.append(ind)
        logger.info("Topographic Wetness Index Generation Completed")
        
        x=grid_x.flatten()
        y=grid_y.flatten()
        geometry_points = [Point(x, y) for x, y in zip(x, y)]
        elevetions = mtop.flatten()
        frame=gpd.GeoDataFrame(columns=["elevation", "geometry"])
        frame['elevation'] = elevetions
        frame['geometry'] = geometry_points
        frame['twi']=slopes
        frame.set_geometry("geometry",inplace=True)
        frame.set_crs("epsg:3395" , inplace=True)
        
        return frame.to_crs(data.crs)

def slope(matrix, res):
    """Generate Slope From Matrix

    Args:
        matrix (np.array): Numpy array representing our matrix
        res (int): Resolution of the given matrix in

    Returns:
        float: Slope of the point in the middle of our matrix
    """
    logger.info("Slope Generation Started")
    if not np.isnan(np.sum(matrix)):
        mat=np.nan_to_num(matrix)
        fx=(mat[2,0] - mat[2,2] +mat[1,0]-mat[1,2]+mat[0,0]-mat[0,2])/(6 *res)
        fy=(mat[0,2]-mat[2,2]+mat[0,1]-mat[2,1]+mat[0,0]-mat[2,0])/(6*res)
        slope=math.atan(((fx**2) + (fy**2))**0.5) 
    
        logger.info("Slope Generation Completed")
        return slope
    else:
        
        logger.exception("Matrix contains nan values: Returning a nan slope")
        return np.nan
    


def slope2(matrix,res):
    """Generate Slope From Matrix

    Args:
        matrix (np.array): Numpy array representing our matrix
        res (int): Resolution of the given matrix in

    Returns:
        float: Slope of the point in the middle of our matrix
    """
    logger.info("Slope Generation Started")
    if not np.isnan(np.sum(matrix)):
        mat=np.nan_to_num(matrix)
        fx=(mat[1,0]-mat[1,2])/(2 *res)
        fy=(mat[0,1]-mat[2,1])/(2*res)
        slope=math.atan(((fx**2) + (fy**2))**0.5) 
        
        logger.info("Slope Generation Completed")
        return slope
    else:
        
        logger.exception("Matrix contains nan values: Returning a nan slope")
        return np.nan

def twi(matrix,res):
    """Generate TWI for Matrix

    Args:
        matrix (array): Array Containing our point and its neighbors
        res (int): Resolution of our matrix

    Returns:
        float: Return TWI for the center point of our matrix
    """
    logger.info("TWI Calculation Started")
    if not np.isnan(np.sum(matrix)):
        mat=np.array(matrix)
        fx=(mat[2,0] - mat[2,2] +mat[1,0]-mat[1,2]+mat[0,0]-mat[0,2])/(6 *res)
        fy=(mat[0,2]-mat[2,2]+mat[0,1]-mat[2,1]+mat[0,0]-mat[2,0])/(6*res)
        slope=((fx**2) + (fy**2))**0.5
        arr=np.array(matrix)
        contributingarea=len(arr[arr>arr[1,1]]) * (res**2)
        if contributingarea==0:
            return 0
        if abs(slope)==0.0:
            return 0
        index=np.log(contributingarea/slope)
        
        logger.info("TWI Calculation Completed")
        return index
    else:
        
        logger.exception("Matrix contains nan values: Returning a nan TWI")
        return 0

def neighbors(mat, row, col, radius=1):
    """Generate neigbors matrix for point

    Args:
        mat (array): Matrix from which the neighbors will be extracted
        row (int): Row of the value for which we are fining the neighbors
        col (int): Col of of the value for which we are fining the neighbors    
        radius (int, optional): Radius of neighbors. Defaults to 1.

    Returns:
        [array]: Returns a matrix of the neighbors with the value of the requred index in the middle 
    """
    
    logger.info("Neighbors Calculation Started")

    rows, cols = len(mat), len(mat[0])
    out = []

    for i in range(row - radius - 1, row + radius):
        row = []
        for j in range(col - radius - 1, col + radius):

            if 0 <= i < rows and 0 <= j < cols:
                try:
                    row.append(mat[i][j])
                except:    
                    logger.exception("Error fininding neighbor")
            else:
                row.append(np.nan)

        out.append(row)
    
    logger.info("Neighbors Calculation Completed")
    return out