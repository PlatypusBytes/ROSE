//Same as ChartModal and AppChart together
<template>
  <v-dialog
    :value="openChartDialog"
    class="chart-dialog"
    max-width="750"
    @click:outside="onClickClose"
  >
    <v-card>
      <v-app-bar flat color="white">
        <v-btn icon @click="onClickClose">
          <v-icon>mdi-close</v-icon>
        </v-btn>
      </v-app-bar>
      <v-divider />
      <v-card-text v-if="hasDataToDisplayInCharts">
        <!-- check if loading is needed -->
        <div class="app-chart">
          <div class="app-chart__canvas">
            <std-chart 
              :mean-values="meanValues" 
              :min-values="minValues"
              :max-values="maxValues"
              :time="time"
            /> 
          </div>
        </div>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<script>
  import StdChart from '~/components/StdChart'
  import { mapState, mapGetters, mapActions } from 'vuex'

  export default { 
    components: { 
      StdChart,
    },
    computed: {
      ...mapGetters('charts', [ 'minValues', 'maxValues', 'meanValues', 'time' ]),
      ...mapState('charts', [ 'openChartDialog' ]),
      hasDataToDisplayInCharts(){
        if (!this.minValues.length 
          || !this.maxValues.length
          || !this.meanValues.length
          || !this.time.length) {
          return false
        }
        return true
      },
    },
    methods: { 
      ...mapActions('charts',[ 'setOpenChartDialog' ]),
      onClickClose() {
        this.setOpenChartDialog(false)
      },
    },
  }

</script>