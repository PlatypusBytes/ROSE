
import getSettlement from '@/lib/requests/get-settlement'
import getDynamicStiffness from '@/lib/requests/get-dynamic-stiffness'
import buildMapboxLayer from '@/lib/mapbox/build-mapbox-layer'
import buildRequestParams from '@/lib/requests/build-request-params'
import buildLayerId from '@/lib/mapbox/build-layer-id'



export default {
  namespaced: true,

  state: () =>( {
    selectedTrainType: null,
    selectedTimeIndex: 0, 
    selectedValueType: null, 
    availableValueTypes: [ { 
                              title: 'Mean',
                              id: 'mean',
                            },{
                              title: 'Standard deviation',
                              id: 'std',
                            },  
                        ],
    selectedLayerType: null,
    mapLayer: null,                   
  }),
  getters: {
    trainTypes(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      return runnerOutput.data.all_train_types
    },
    colors(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      return runnerOutput.data.colours
    },
    cummulativeSetLimits(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      const { selectedTimeIndex } = state

      return runnerOutput.data.cumulative_sett_limits[selectedTimeIndex]
    },
    cummulativeSetStdLimits(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      const { selectedTimeIndex } = state
   
      return runnerOutput.data.cumulative_sett_std_limits[selectedTimeIndex]
    },
    dynamicStiffnessLimits(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      return runnerOutput.data.dyn_stiff_limits
    },
    dynamicStiffnessStdLimits(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      return runnerOutput.data.dyn_stiff_std_limits
    },
    legend(state, getters) {
      const { selectedLayerType, selectedValueType } = state
      
      const { cummulativeSetLimits, cummulativeSetStdLimits,
         dynamicStiffnessLimits, dynamicStiffnessStdLimits, colors } = getters
      let legend = []
      let limits = []
      if (selectedLayerType === 'settlement') {
      limits = selectedValueType === 'mean' ? cummulativeSetLimits : cummulativeSetStdLimits
      }else{
        limits = selectedValueType === 'mean' ? dynamicStiffnessLimits : dynamicStiffnessStdLimits
      }
      colors.forEach((color, index) => {
              legend.push({ color: color, label : limits[index].toString() }) 
        })
    return legend
    },
    legendTitle(state) {
      const { selectedLayerType, selectedValueType } = state
      let title = ''
      if (selectedLayerType === 'settlement') {
        title = selectedValueType === 'mean' ? 'Mean settlement [mm]' : 'Std settlement [mm]'
      }else{
        title = selectedValueType === 'mean' ? 'Mean dynamic stiffness [kN/m]' : 'Std dynamic stiffness [kN/m]'
      }

      return title
    },
    timeIndexes(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      return runnerOutput.data.time
    }, 
    layerId(state) {
      const { selectedLayerType, selectedValueType, selectedTrainType, selectedTimeIndex } = state
      return buildLayerId(selectedLayerType, selectedValueType, selectedTrainType, selectedTimeIndex)
    },
    params(state) {
      const { selectedLayerType, selectedTimeIndex, selectedTrainType, selectedValueType } = state
      return buildRequestParams(selectedLayerType, selectedTimeIndex, selectedTrainType, selectedValueType)
      },
  },
  actions: {
    setTrainType(context, payload) {
      context.commit('SET_TRAIN_TYPE', payload)
    },
    setTimeIndex(context, payload) {
      context.commit('SET_TIME_INDEX', payload)
    },
    setValueType(context, payload) {
      context.commit('SET_VALUE_TYPE', payload)
    },
    setLayerType(context, payload) {
      context.commit('SET_LAYER_TYPE', payload)
    },
    async getFeaturesCollection(context) {
      const { selectedLayerType, selectedValueType } = context.state
      
      const { params, colors,  cummulativeSetLimits, 
              dynamicStiffnessLimits, cummulativeSetStdLimits, dynamicStiffnessStdLimits } = context.getters
      const emptyParams = Object.values(params).some(param => param === null )
      if (emptyParams) {
        return 
      }
      if (selectedLayerType && selectedLayerType === 'settlement') {
        const featuresCollection =  await getSettlement(params)
        const { layerId } = context.getters

        const mapLayer = selectedValueType === 'mean' ? buildMapboxLayer(layerId, featuresCollection, colors, cummulativeSetLimits ) 
                                                      : buildMapboxLayer(layerId, featuresCollection, colors, cummulativeSetStdLimits ) 
        context.commit('SET_MAP_LAYER', mapLayer)

      }else{

        const featuresCollection =  await getDynamicStiffness(params)
        const { layerId } = context.getters
        
        const mapLayer = selectedValueType === 'mean' ? buildMapboxLayer(layerId, featuresCollection, colors, dynamicStiffnessLimits ) 
                                                      : buildMapboxLayer(layerId, featuresCollection, colors, dynamicStiffnessStdLimits ) 
        context.commit('SET_MAP_LAYER', mapLayer)
      }
    },  
  },
  mutations:{
    SET_TRAIN_TYPE(state, type) {
      state.selectedTrainType = type
    },
    SET_TIME_INDEX(state, index) {
      state.selectedTimeIndex = index
    },
    SET_VALUE_TYPE(state, type) {
      state.selectedValueType = type
    },
    SET_LAYER_TYPE(state, type) {
      state.selectedLayerType = type
    },
    SET_MAP_LAYER(state, layer) {
      state.mapLayer = layer
    },
  },

}