export default(layerType, valueType, trainType, timeIndex) => {
  if (!trainType) {
    return null
  }
  return layerType === 'settlement' ? `${ valueType }_${ timeIndex }` : `${ valueType }_${ trainType }` 
}