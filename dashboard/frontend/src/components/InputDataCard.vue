<template>
  <v-container fluid>
    <v-card
      class="mx-auto my-12"
      max-width="374"
      height="500"
    >
      <v-card-title>
        Input Data
      </v-card-title>
      <v-card-text>
        <v-file-input 
          label="Import SOS segments"
          hint="Compulsory"
          prepend-icon=""
          required
          @change="onSOSinput"
        />
        <v-file-input
          label="Import Sensar (Optional)"
          prepend-icon=""
          hint="Optional"
          @change="onSensarInput"
        />
        <v-file-input
          label="Import Rila (Optional)"
          prepend-icon=""
          hint="Optional"
          @change="onRilaInput"
        />
        <v-file-input
          label="Import InfraMon (Optional)"
          prepend-icon=""
          hint="Optional"
          @change="onInfraMonInput"
        />
      </v-card-text>
      <v-card-actions>
        <v-container>
          <v-row dense>
            <div class="flex-grow-1" />
            <v-btn
              color="primary"
              :disabled="disableRun"
              @click="run"
            >
              RUN
            </v-btn>
          </v-row>
          <v-row v-if="warningMessage">
            <v-col cols="12">
              <v-alert type="error">
                {{ warningMessage }}
              </v-alert>
            </v-col>
          </v-row>
        </v-container>
      </v-card-actions>
    </v-card>
  </v-container>
</template>
<script>
  import { mapActions, mapGetters } from 'vuex'

  export default {
    data() {
      return {
        warningMessage: null,
      }
    },
    computed: {
      ...mapGetters('inputs', [ 'runnerInput', 'inputValid', 'message', 'running' ]),

      disableRun() {
        if (!this.runnerInput) {
          return true
        }else return false
      },
    },
    watch: {
      inputValid() {
        this.showErrorMessage()
        this.goToResultsPage()
      },
      running() {
        this.showRunningMessage() 
      },
    },

    methods: {
      ...mapActions('inputs', [ 'setSosSegmentInput','setSensarInput', 'setRilaInput',  'setInfraMonInput', 'startRunner' ]),
      onSOSinput(event) {
        this.setSosSegmentInput(event)
      },
      onSensarInput(event) {
        this.setSensarInput(event)
      },
      onRilaInput(event) {
        this.setRilaInput(event)
      },
      onInfraMonInput(event) {
        this.setInfraMonInput(event)
      },
      run() {
        this.startRunner()
      },
      goToResultsPage() {
        const valid = this.inputValid
        const running = this.running
      
        if (valid) {
          (valid === true && running === false) ? this.$emit('go-to-results', 2) : null
        }
      },
      showErrorMessage() {
        const valid = this.inputValid
        valid === false ? this.warningMessage = this.message : null
        
      },
      showRunningMessage() { 
        const running = this.running
        running === true ? this.warningMessage = this.message : null
      },
    },

  }

</script>
