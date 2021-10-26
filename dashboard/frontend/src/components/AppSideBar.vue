<template>
  <div class="pa-4 home d-flex flex-column">
    <v-expansion-panels v-model="panel" class="mb-4">
      <content-panels 
        id="dyn_stiffness" 
        title="Dynamic Stiffness" 
      > 
        <train-type-list  
          slot="trainTypes"
          :train-types="trainTypes"
          @set-train-type="setTrainType"
        />
        <value-type-selection
          slot="valueTypes"
          :value-types="availableValueTypes"
          @set-value-type="setValueType"
        />
      </content-panels>
      <content-panels id="settlement" title="Settlement"> 
        <time-slider 
          slot="slider" 
          :time-indexes="timeIndexes"
          @set-time-index="setTimeIndex"
        />
        <value-type-selection 
          slot="valueTypes"
          :value-types="availableValueTypes"
          @set-value-type="setValueType"
        />
      </content-panels>
    </v-expansion-panels>
  </div>
</template>

<script>
  import ContentPanels from '~/components/ContentPanels'
  import ValueTypeSelection from '~/components/ValueTypeSelection'
  import TrainTypeList from '~/components/TrainTypeList'
  import TimeSlider from '~/components/TimeSlider'
  import { mapState, mapGetters, mapActions } from 'vuex'

  export default {
    components: {
      ContentPanels,
      TrainTypeList,
      TimeSlider,
      ValueTypeSelection,
    },
    data() { 
    
      return {
        panel: 0,
      }
    },
    computed: { 
      ...mapState('layers', [ 'availableValueTypes' ]),
      ...mapGetters('layers', [ 'timeIndexes', 'trainTypes','params' ]),
    },
    watch: { 
      params:{
        handler(){
          this.getFeaturesCollection()
        },
      },
    },
    methods: {
      ...mapActions('layers', [ 'setTrainType', 'setTimeIndex', 'setValueType', 'getFeaturesCollection' ]),
    },
  }


</script>

  