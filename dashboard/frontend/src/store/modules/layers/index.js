
import getSettlement from '@/lib/requests/get-settlement'
import getDynamicStiffness from '@/lib/requests/get-dynamic-stiffness'
import buildMapboxLayer from '@/lib/mapbox/build-mapbox-layer'
import buildRequestParams from '@/lib/requests/build-request-params'
import buildLayerId from '@/lib/mapbox/build-layer-id'
import mean_stiffness_intercity from '~/data/mean_stiffness_intercity.json'
import mean_settlement_25 from '~/data/mean_settlement_25.json'

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

        //const feat = mean_settlement_25 //TODO remove the sample test
        context.commit('SET_MAP_LAYER', buildMapboxLayer(layerId, featuresCollection, colors, cummulativeSetLimits ))

      }else{
        const featuresCollection =  await getDynamicStiffness(params)
        const { layerId } = context.getters
        
        //const feat = mean_stiffness_intercity //TODO remove the sample test
        context.commit('SET_MAP_LAYER', buildMapboxLayer(layerId, featuresCollection, colors, dynamicStiffnessLimits ))
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