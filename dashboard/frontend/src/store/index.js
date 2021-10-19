import Vue from 'vue'
import Vuex from 'vuex'

import layers from './modules/layers'
import inputs from './modules/inputs'
import charts from './modules/charts'

Vue.use(Vuex)

export default new Vuex.Store({
  actions: {},
  modules: {
    layers,
    inputs,
    charts,
  },
})