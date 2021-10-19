
import getSettlement from '@/lib/requests/get-settlement'
import getDynamicStiffness from '@/lib/requests/get-dynamic-stiffness'
import buildMapboxLayer from '@/lib/mapbox/build-mapbox-layer'
import buildRequestParams from '@/lib/requests/build-request-params'
import buildLayerId from '@/lib/mapbox/build-layer-id'
import sample from '~/data/sample.json'

export default {
  namespaced: true,

  state: () =>( {
    selectedTrainType: null,
    selectedTimeIndex: null, 
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
      return runnerOutput.data.cumulative_sett_limits
    },
    dynamicStiffnessLimits(state, getters, rootState) {
      const { runnerOutput } = rootState.inputs
      if (!runnerOutput) {
        return []
      }
      return runnerOutput.data.dyn_stiff_limits
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
      const { selectedLayerType } = context.state
      const { params, colors,  cummulativeSetLimits, dynamicStiffnessLimits } = context.getters
      if (selectedLayerType && selectedLayerType === 'settlement') {
        const featuresCollection =  await getSettlement(params)
        const { layerId } = context.getters
        context.commit('SET_MAP_LAYER', buildMapboxLayer(layerId, featuresCollection, colors, cummulativeSetLimits ))
      }else{
        const featuresCollection =  await getDynamicStiffness(params)
        const { layerId } = context.getters
        const feat = sample
        
        context.commit('SET_MAP_LAYER', buildMapboxLayer(layerId, feat, colors, dynamicStiffnessLimits ))
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