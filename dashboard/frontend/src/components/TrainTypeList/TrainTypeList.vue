<template>
  <div class="pa-4">
    <div class="text-body-1 pb-3">
      Train type:
    </div>
    <v-select
      v-model="selectedType"
      :items="trainTypes"
      outlined
      desnse
    />
  </div>
</template>

<script>
  export default {   
    props: {
      trainTypes: {
        type: Array,
        required: true,
        default: () => [],
      },
      
    },
    data() { 
      return { 
        selectedType: null,
      }
    },
    watch: {
      selectedType() {
        this.setTrainType()
      },
    },
    mounted() { 
      this.selectedType = this.trainTypes[0]
    },
    destroyed() {
      this.setTrainType(null)
    },
    methods: {
      setTrainType() {
        this.$emit('set-train-type', this.selectedType)
      },
    },
  }
</script>