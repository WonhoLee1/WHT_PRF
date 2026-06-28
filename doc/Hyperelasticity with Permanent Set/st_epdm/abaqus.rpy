# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2020.HF3 replay file
# Internal Version: 2020_05_13-03.40.25 163480
# Run by goodman on Sun May 16 20:15:03 2021
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=266.919769287109, 
    height=190.400009155273)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='st_epdm.odb')
#: Model: E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       2
#: Number of Node Sets:          8
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    DEFORMED, ))
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Reaction force: RF1 at Node 1 in NSET NRF1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
c1 = session.Curve(xyData=xy1)
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Spatial displacement: U1 at Node 2 in NSET NU1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
c1 = session.Curve(xyData=xy1)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Spatial displacement: U1 at Node 2 in NSET NU1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
c1 = session.Curve(xyData=xy1)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Reaction force: RF1 at Node 1 in NSET NRF1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
c1 = session.Curve(xyData=xy1)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy0 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Reaction force: RF1 at Node 1 in NSET NRF1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Spatial displacement: U1 at Node 2 in NSET NU1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
xy2 = combine(xy0, xy1, )
xy_result = session.XYData(name='XYData-1', objectToCopy=xy2, 
    sourceDescription='combine(XYData-1, XYData-1, )')
del session.xyDataObjects[xy0.name]
del session.xyDataObjects[xy1.name]
del session.xyDataObjects[xy2.name]
c1 = session.Curve(xyData=xy_result)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy0 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Reaction force: RF1 at Node 1 in NSET NRF1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Spatial displacement: U1 at Node 2 in NSET NU1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
xy2 = combine(xy0, xy1, )
xy_result = session.XYData(name='XYData-2', objectToCopy=xy2, 
    sourceDescription='combine(XYData-2, XYData-2, )')
del session.xyDataObjects[xy0.name]
del session.xyDataObjects[xy1.name]
del session.xyDataObjects[xy2.name]
c1 = session.Curve(xyData=xy_result)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Spatial displacement: U1 at Node 2 in NSET NU1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
c1 = session.Curve(xyData=xy1)
xy2 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Reaction force: RF1 at Node 1 in NSET NRF1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
c2 = session.Curve(xyData=xy2)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, c2, ), appendMode=True)
session.charts[chartName].autoColor(lines=True, symbols=True)
odb = session.odbs['E:/Simulia_Video_Contents/Material Calibration/Hyperelasticity with Permanent Set/st_epdm/st_epdm.odb']
xy0 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Spatial displacement: U1 at Node 2 in NSET NU1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
xy1 = xyPlot.XYDataFromHistory(odb=odb, 
    outputVariableName='Reaction force: RF1 at Node 1 in NSET NRF1', 
    suppressQuery=True, __linkedVpName__='Viewport: 1')
xy2 = combine(xy0, xy1, )
xy_result = session.XYData(name='XYData-3', objectToCopy=xy2, 
    sourceDescription='combine(XYData-3, XYData-3, )')
del session.xyDataObjects[xy0.name]
del session.xyDataObjects[xy1.name]
del session.xyDataObjects[xy2.name]
c1 = session.Curve(xyData=xy_result)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
