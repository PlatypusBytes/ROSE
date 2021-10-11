<template>
  <v-container fluid>
    <v-card
      class="mx-auto my-12"
      max-width="374"
      height="420"
    >
      <v-card-title>
        Input Data
      </v-card-title>
      <v-card-text>
        <v-file-input 
          label="Import SOS segments"
          hint="Compulsory"
          prepend-icon=""
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
        <div class="flex-grow-1" />
        <v-btn
          color="primary"
          :disabled="disableRun"
          @click="run"
        >
          RUN
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>
<script>
  import { mapActions, mapGetters } from 'vuex'

  export default {
    computed: {
      ...mapGetters([ 'runnerInput' ]),
      disableRun() {
        if (!this.runnerInput) {
          return true
        }else return false
      },
    },

    methods: {
      ...mapActions([ 'setSosSegmentInput','setSensarInput', 'setRilaInput',  'setInfraMonInput', 'startRunner' ]),
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
        //this.$emit('go-to-results', 2)
      },
    },

  }

</script>
