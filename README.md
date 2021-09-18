

<!--
*** Adapted from the best readme template here https://github.com/othneildrew/Best-README-Template/blob/master/README.md
-->



<!-- PROJECT SHIELDS -->



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">AgriTech</h1>

  <p align="center">
    Module to fetch, transform and visualize 3dep lidar data
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#libraries-used">Libraries Used</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


The aim of this prodject is to produce an easy to use, reliable and well designed python module that domain experts and data scientists can use to fetch, visualise, and transform publicly available satellite and LIDAR data. Module will interface with USGS 3DEP and fetch data using their API.

### Built With
* [Python](https://www.python.org/)
### Libraries used 
* [PDAL](https://pypi.org/project/PDAL/)
* [scipy](https://scipy.org/)
* [shapely](https://pypi.org/project/Shapely)
* [geopandas](https://geopandas.org/)



<!-- GETTING STARTED -->
## Getting Started

How to setup the module

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/SameC137/AgriTech
   ```
3. Move to the package directory
   ```sh
   cd elevation_3DEP
   ```
4. Install with pip
   ```sh
   pip install .
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

<b>Import</b>
```
from eleveation_3DEP.fetch import FetchData
```
<b>Getting Data</b>
```python
FetchData(region).fetch_elevation(geometry,csr)
```
returns a dataframe containeing elevation for points inside geometry
|	|elevation	|geometry
|----------|:-------------:|------:|
|0	|310.10| POINT  (-93.75605 41.91804)
|1	|310.56	|POINT (-93.75566 41.91819)
|2	|310.40	|POINT (-93.75605 41.91864)
|3	|311.59 |POINT (-93.75606 41.91908)
|4	|312.16	|POINT (-93.75586 41.91923)

<b>Visualization in 3D</b>
```python
FetchData(region).visualize3D(dataframe)
```

<img src="https://drive.google.com/uc?export=view&id=1kLw_bOgHKESyDMMuQuNAAZffo1SQ6Ati"/>


<b>Visualization in 2D</b>
```python
FetchData(region).visualize2D(dataframe)
```
<img src="https://drive.google.com/uc?export=view&id=1uMz0UYXiq1RILhrdomzSrvm0bLHY3GLG">

<b>Standardize into grid</b>
```python
FetchData(region).standardize(dataframe,resoulution)
```
<b>Topographic wetness index</b>
```python
FetchData(region).topographicWetnessIndex(dataframe,resolution)
```
For more a full list of functionality and parameters, please refer to the [Documentation](https://github.com/SameC137/AgriTech/Documentation)
or checkout the notebook demonstration [Demonstration](https://github.com/SameC137/AgriTech/notebooks/Demonstrations.ipynb)


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/SameC137/AgriTech/issues) for a list of proposed features (and known issues).




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.




