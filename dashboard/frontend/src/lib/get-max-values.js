export default (meanValues, stdValues) => {
  if (!meanValues || !stdValues) {
    return []
  }
  const maxValues = meanValues.map((value, index) => {
    const max = value + stdValues[index]
    return max
  })
  return maxValues
}