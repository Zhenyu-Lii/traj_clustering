import transbigdata as tbd
import pandas as pd
import os
import numpy as np
import geopandas as gpd
from transbigdata.grids import (
    area_to_params,
    GPS_to_grid,
    grid_to_polygon,
    grid_to_centre,
)
from transbigdata.odprocess import (
    odagg_grid
)

def visualization_data(data, col=['lon', 'lat'], accuracy=500, height=500,
                       maptype='point', zoom='auto'):
    '''
    The input is the data points, this function will aggregate and then
    visualize it

    Parameters
    -------
    data : DataFrame
        The data point
    col : List
        The column name. The user can choose a non-weight Origin-Destination
        (OD) data, in the sequence of [longitude, latitude]. For this, The
        aggregation is automatic. Or, the user can also input a weighted OD
        data, in the sequence of [longitude, latitude, count]
    zoom : number
        Map zoom level (Optional). Default value: auto
    height : number
        The height of the map frame
    accuracy : number
        Grid size
    maptype : str
        Map type, ‘point’ or ‘heatmap’

    Returns
    -------
    vmap : keplergl.keplergl.KeplerGl
        Visualizations provided by keplergl
    '''
    try:
        from keplergl import KeplerGl
    except ImportError: # pragma: no cover
        raise ImportError( # pragma: no cover
            "Please install keplergl, run "
            "the following code in cmd: pip install keplergl")

    if len(col) == 2:
        lon, lat = col[0], col[1]
        count = 'count'

        data[lon] = data[lon].astype('float')
        data[lat] = data[lat].astype('float')
        #clean data
        data = data[-((data[lon].isnull())|(data[lat].isnull()))]
        data = data[(data[lon]>=-180)&(data[lon]<=180)&(data[lat]>=-90)&(data[lat]<=90)]

        bounds = [data[lon].min(), data[lat].min(),
                  data[lon].max(), data[lat].max()]
        lon_center, lat_center = data[lon].mean(), data[lat].mean()
        if zoom == 'auto':
            lon_min, lon_max = data[lon].quantile(
                0.05), data[lon].quantile(0.95)
            zoom = 8.5-np.log(lon_max-lon_min)/np.log(2)
        params = area_to_params(bounds, accuracy=accuracy)
        data['LONCOL'], data['LATCOL'] = GPS_to_grid(
            data[lon], data[lat], params)
        data[count] = 1
        data = data.groupby(['LONCOL', 'LATCOL'])[
            'count'].sum().reset_index().reset_index()
        data['geometry'] = grid_to_polygon(
            [data['LONCOL'], data['LATCOL']], params)
        data[lon], data[lat] = grid_to_centre(
            [data['LONCOL'], data['LATCOL']], params)
        data = gpd.GeoDataFrame(data)
        data.to_file('current_data.geojson', driver='GeoJSON')

    if len(col) == 3:
        lon, lat, count = col

        data[lon] = data[lon].astype('float')
        data[lat] = data[lat].astype('float')
        #clean data
        data = data[-((data[lon].isnull())|(data[lat].isnull()))]
        data = data[(data[lon]>=-180)&(data[lon]<=180)&(data[lat]>=-90)&(data[lat]<=90)]

        bounds = [data[lon].min(), data[lat].min(),
                  data[lon].max(), data[lat].max()]
        lon_center, lat_center = data[lon].mean(), data[lat].mean()
        if zoom == 'auto':
            lon_min, lon_max = data[lon].quantile(
                0.05), data[lon].quantile(0.95)
            zoom = 8.5-np.log(lon_max-lon_min)/np.log(2)
        params = area_to_params(bounds, accuracy=accuracy)
        data['LONCOL'], data['LATCOL'] = GPS_to_grid(
            data[lon], data[lat], params)
        data = data.groupby(['LONCOL', 'LATCOL'])[count].sum().reset_index()
        data['geometry'] = grid_to_polygon(
            [data['LONCOL'], data['LATCOL']], params)
        data[lon], data[lat] = grid_to_centre(
            [data['LONCOL'], data['LATCOL']], params)

        data = gpd.GeoDataFrame(data)

    if maptype == 'heatmap':
        vmap = KeplerGl(config={ # pragma: no cover
            'version': 'v1',
            'config': {
                'visState': {
                    'filters': [],
                    'layers': [
                        {'id': 'vpefba0o',
                         'type': 'heatmap',
                         'config': {
                             'dataId': 'data',
                             'label': 'Point',
                             'color': [18, 147, 154],
                             'highlightColor': [252, 242, 26, 255],
                             'columns': {'lat': lat, 'lng': lon},
                             'isVisible': True,
                             'visConfig': {
                                 'opacity': 0.8,
                                 'colorRange': {
                                     'name': 'Global Warming',
                                     'type': 'sequential',
                                     'category': 'Uber',
                                     'colors': ['#5A1846',
                                                '#900C3F',
                                                '#C70039',
                                                '#E3611C',
                                                '#F1920E',
                                                '#FFC300']},
                                 'radius': 23},
                             'hidden': False,
                             'textLabel': [{'field': None,
                                            'color': [255, 255, 255],
                                            'size': 18,
                                            'offset': [0, 0],
                                            'anchor': 'start',
                                            'alignment': 'center'}]},
                         'visualChannels': {
                             'weightField': {'name': count, 'type': 'integer'},
                             'weightScale': 'linear'}}],
                    'interactionConfig': {
                        'tooltip': {
                            'fieldsToShow': {
                                'data': [{
                                    'name': count,
                                    'format': None}]},
                            'compareMode': False,
                            'compareType': 'absolute',
                            'enabled': True},
                        'brush': {'size': 0.5, 'enabled': False},
                        'geocoder': {'enabled': False},
                        'coordinate': {'enabled': False}},
                    'layerBlending': 'normal',
                    'splitMaps': [],
                    'animationConfig': {'currentTime': None, 'speed': 1}},
                'mapState':
                {
                    'bearing': 0,
                    'dragRotate': True,
                    'latitude': lat_center,
                    'longitude': lon_center,
                    'pitch': 0,
                    'zoom': zoom,
                    'isSplit': False
                },
                'mapStyle':
                {
                    'styleType': 'dark',
                    'topLayerGroups':
                    {},
                    'visibleLayerGroups':
                    {
                        'label': True,
                        'road': True,
                        'border': False,
                        'building': True,
                        'water': True,
                        'land': True,
                        '3d building': False
                    },
                    'threeDBuildingColor': [9.665468314072013,
                                            17.18305478057247,
                                            31.1442867897876
                                            ],
                    'mapStyles':
                    {}
                }}}, data={'data': data.to_json()}, height=height)
    else:
        vmap = KeplerGl(config={
            'version': 'v1',
            'config': {
                'visState': {
                    'filters': [],
                    'layers': [{
                        'id': 'ytak0zp',
                        'type': 'geojson',
                        'config': {
                            'dataId': count,
                            'label': count,
                            'color': [77, 193, 156],
                            'highlightColor': [252, 242, 26, 255],
                            'columns': {'geojson': '_geojson'},
                            'isVisible': True,
                            'visConfig': {
                                'opacity': 0.8,
                                'strokeOpacity': 0.8,
                                'thickness': 0.5,
                                'strokeColor': [218, 112, 191],
                                'colorRange': {
                                    'name': 'Global Warming',
                                    'type': 'sequential',
                                    'category': 'Uber',
                                    'colors': ['#5A1846',
                                               '#900C3F',
                                               '#C70039',
                                               '#E3611C',
                                               '#F1920E',
                                               '#FFC300']},
                                'strokeColorRange': {'name': 'Global Warming',
                                                     'type': 'sequential',
                                                     'category': 'Uber',
                                                     'colors': ['#5A1846',
                                                                '#900C3F',
                                                                '#C70039',
                                                                '#E3611C',
                                                                '#F1920E',
                                                                '#FFC300']},
                                'radius': 10,
                                'sizeRange': [0, 10],
                                'radiusRange': [0, 50],
                                'heightRange': [0, 500],
                                'elevationScale': 5,
                                'enableElevationZoomFactor': True,
                                'stroked': False,
                                'filled': True,
                                'enable3d': False,
                                'wireframe': False},
                            'hidden': False,
                            'textLabel': [{'field': None,
                                           'color': [255, 255, 255],
                                           'size': 18,
                                           'offset': [0, 0],
                                           'anchor': 'start',
                                           'alignment': 'center'}]},
                        'visualChannels': {
                            'colorField': {'name': count, 'type': 'integer'},
                            'colorScale': 'quantile',
                            'strokeColorField': None,
                            'strokeColorScale': 'quantile',
                            'sizeField': None,
                            'sizeScale': 'linear',
                            'heightField': None,
                            'heightScale': 'linear',
                            'radiusField': None,
                            'radiusScale': 'linear'}}],
                    'layerBlending': 'normal',
                    'splitMaps': [],
                    'animationConfig': {'currentTime': None, 'speed': 1}},
                'mapState': {'bearing': 0,
                             'dragRotate': False,
                             'latitude': data[lat].mean(),
                             'longitude': data[lon].mean(),
                             'pitch': 0,
                             'zoom': 10,
                             'isSplit': False}}},
            data={count: data.to_json()}, height=height)

    return vmap

file_path = os.path.abspath('..') + '/preprocess/'

df_all = pd.read_hdf('../preprocess/Geolife_all_user.h5',key='data')

data = df_all[df_all['user']==0]

visualization_data(data,col=['lon', 'lat'],accuracy=10)