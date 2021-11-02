export default (meanValues, stdValues) => {
  if (!meanValues || !stdValues) {
    return []
  }
  const minValues = meanValues.map((value, index) => {
    const min = value - stdValues[index]
    return min
  })
  return minValues
}
