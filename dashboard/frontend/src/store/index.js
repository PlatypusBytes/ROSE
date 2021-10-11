import Vue from 'vue'
import Vuex from 'vuex'
import runner from '@/lib/requests/runner'
import openFile from '@/lib/load-file'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    runnerOutput: null,
    sosSegmentsInput: null,
    sensarInput: null,
    rilaInput: null,
    infraMonInput: null,

  },
  getters: {
    runnerInput(state) {
      const { sosSegmentsInput, sensarInput, rilaInput, infraMonInput } = state
      if (sosSegmentsInput) {
        return {
          'SOS_Segment_Input': sosSegmentsInput,
          'Sensar_Input': sensarInput,
          'Rila_Input': rilaInput,
          'InfraMon_Input': infraMonInput,
        }
      } 
    },
  },
  mutations: {
    SET_RUNNER_OUTPUT(state, response) {
      state.runnerOutput = response
    },
    SET_SOS_SEGMENT_INPUT(state, input) {
      state.sosSegmentsInput = input
    },
    SET_SENSAR_INPUT(state, input) {
      state.sensarInput = input
    },
    SET_RILA_INPUT(state, input) {
      state.rilaInput = input
    },
    SET_INFRA_MON_INPUT(state, input) {
      state.infraMonInput = input
    },
  },
  actions: {
    async startRunner({ commit, state, getters }) {
      const  input  = getters.runnerInput
      response = await runner(input)
      console.log('response', response)
    },
    async setSosSegmentInput(context, payload) {
      if (payload) {
        const input = await openFile(payload)
        context.commit('SET_SOS_SEGMENT_INPUT', input)
      }else {
        context.commit('SET_SOS_SEGMENT_INPUT', null)
      }
     
    },
    async setSensarInput(context, payload) {
      if (payload) {
        const input = await openFile(payload)
        context.commit('SET_SENSAR_INPUT', input)
      }else {
        context.commit('SET_SENSAR_INPUT', null)
      }
    },
    async setRilaInput(context, payload) {
      if (payload) {
        const input = await openFile(payload)
        context.commit('SET_RILA_INPUT', input)
      }else{
         context.commit('SET_RILA_INPUT', null)
      }
  
    },
    async setInfraMonInput(context, payload) {
      if (payload) {
        const input = await openFile(payload)
        context.commit('SET_INFRA_MON_INPUT', input)       
      }else{
        context.commit('SET_INFRA_MON_INPUT', null)   
      }
    },

  },
})

