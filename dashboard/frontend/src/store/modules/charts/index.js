import getChartData from '@/lib/requests/get-chart-data'
import getMinValues from '@/lib/get-min-values'
import getMaxValues from '@/lib/get-max-values'

 

export default {
  namespaced: true,
  state: () => ({
    chartData: null,
    openChartDialog: false,
  }),
  getters: {
    meanValues(state) {
      const { chartData } = state
      if (!chartData) {
        return []
      }
      return chartData.cumulative_settlement_mean
    },
    stdValues(state) {
      const { chartData } = state
      if (!chartData) {
        return []
      }
      return chartData.cumulative_settlement_std
    },
    time(state) {
      const { chartData } = state
      if (!chartData) {
        return []
      }
      return chartData.time 
    }, 
    minValues(state, getters) {
      const { meanValues, stdValues } = getters
      return getMinValues(meanValues, stdValues)
    },
    maxValues(state, getters) {
      const { meanValues, stdValues } = getters
      return getMaxValues(meanValues, stdValues)
    },
  },
  actions: {
    async setChartData(context, payload){
     
      const data = await getChartData(payload) 

      context.commit('SET_CHART_DATA', data)
      context.commit('SET_OPEN_CHART_DIALOG', true)
    },
    setOpenChartDialog(context, payload) {
      context.commit('SET_OPEN_CHART_DIALOG', payload)
    },
  },
  mutations: {
    SET_CHART_DATA(state, data) {
      state.chartData = data
    },
    SET_OPEN_CHART_DIALOG(state, boolean) {
      state.openChartDialog = boolean
    },
  },
}