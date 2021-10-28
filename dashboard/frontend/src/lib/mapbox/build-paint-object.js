  
  export default(colors, limits) => {

    let colorsLimits = [ ]

    colors.forEach((color, index)=> {
      colorsLimits = [ ...colorsLimits, color, limits[index] ]
    })
    colorsLimits.push('#000066')
    return {
      'line-width' : 3,
      'line-color': [
        'step', [ 'get', 'value' ], ...colorsLimits,
      ],
    }
  }
  