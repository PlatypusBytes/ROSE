export default {
  namespaced: true,
  state: () => ({}),
  getters: {},
  acttions: {
    getChartData(){
      console.log('call to API for chart data')
    },
  },
}