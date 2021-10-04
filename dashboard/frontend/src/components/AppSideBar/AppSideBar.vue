//Move from App.vue the navigation-bar
// Import the collapbsibleCard
// Import the layers 
// Import the timeslider 
// Import the combobox
<template>
  <div class="pa-4 home d-flex flex-column">
    <collapbsible-card title="Dynamic Stiffness"> 
      <train-type-list slot="select" />
      <layer-group slot="layers" />
    </collapbsible-card>
    <collapbsible-card title="Settlement"> 
      <time-slider slot="slider" />
      <layer-group slot="layers" />
    </collapbsible-card>
  </div>
</template>

<script>
  import CollapbsibleCard from '~/components/CollapsibleCard/CollapbsibleCard'
  import LayerGroup from '~/components/LayerGroup/LayerGroup'
  import TrainTypeList from '~/components/TrainTypeList/TrainTypeList'
  import TimeSlider from '~/components/TimeSlider/TimeSlider'

  export default {
    components: {
      CollapbsibleCard,
      LayerGroup,
      TrainTypeList,
      TimeSlider,
    },


  }


</script>

  