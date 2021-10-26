import buildPaintObject from './build-paint-object'
export default ( id, data, colors, limits ) => {
 
  return {
    id,
    type: 'line',
    source: {
      type: 'geojson',
      data,
    },
    paint: buildPaintObject(colors, limits),
  }
}
