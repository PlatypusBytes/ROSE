<template>
  <v-radio-group
    v-model="selectedType"
    class="pa-4"
  >
    <v-radio
      v-for="valueType in valueTypes"
      :key="valueType.id"
      :label="valueType.title"
      :value="valueType.id"
    />
  </v-radio-group>
</template>

<script>
  export default {
    props: {
      valueTypes: {
        type: Array,
        required: true,
        default: () => [],
      },
    },
    data() {
      return { 
        selectedType: this.valueTypes[0].id,
      }
    },
    watch: {
      selectedType(){
        this.setSelectedType()
      },
    },
    mounted() { 
      this.setSelectedType()
    },
    methods: {
      setSelectedType() {
        this.$emit('set-value-type', this.selectedType)
      },
    },
  }

</script>
