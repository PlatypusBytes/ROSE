
import runner from '@/lib/requests/runner'
import openFile from '@/lib/load-file'


export default {
  namespaced: true,

  state: () =>({
    runnerOutput: null,
    sosSegmentsInput: null,
    sensarInput: null,
    rilaInput: null,
    infraMonInput: null,

  }),
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
    dataExist(state) {
      const { runnerOutput } = state
      if (!runnerOutput) {
        return null
      }
      return runnerOutput.exist
    },
    inputValid(state) {
      const { runnerOutput } = state
      if (!runnerOutput) {
        return null
      }
      //return runnerOutput.valid  
      return true
    },
    message(state) {
     const { runnerOutput } = state
      if (!runnerOutput) {
        return null
      }
      return runnerOutput.message 
    },
  },
  actions: {
    async startRunner(context) {
      const  input  = context.getters.runnerInput
      const response = await runner(input)
      context.commit('SET_RUNNER_OUTPUT', response) 
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

}

