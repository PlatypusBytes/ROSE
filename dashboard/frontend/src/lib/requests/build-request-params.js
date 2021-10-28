/* Create the parameters object for the different type of requests (dynamic stiffness or  cumulative settlement) */
export default((layerType, timeIndex, trainType, valueType) => {
      if (layerType === 'settlement') {
        return {
          time_index: timeIndex,
          value_type: `cumulative_settlement_${ valueType }`,
        }} else if (layerType === 'dyn_stiffness') {
            return {
              train_type: trainType,
              value_type: `${ valueType }_dyn_stiffness`,
            }
        }
      return null
})

