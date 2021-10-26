<template>
  <div class="pa-4  mt-5">
    <div class="text-body-1 pb-3">
      Time:
    </div>
    <v-slider 
      v-model="selectedIndex"
      :thumb-label="true"
      :min="timeIndexes[0]"
      :max="timeIndexes[timeIndexes.length -1]"
    />
  </div>
</template>

<script>
 
  export default {
    props: {
      timeIndexes: {
        type: Array,
        required: true,
        default: () => [],
      },
    },
    data() { 
      return { 
        selectedIndex: null,
      }
    },
    watch: { 
      selectedIndex(){
        this.setTimeIndex()
      },
    },
    mounted() { 
      this.selectedIndex = this.timeIndexes[0]
    },
    methods: { 
      setTimeIndex() {
        this.$emit('set-time-index', this.selectedIndex)
      },
    },
  }
</script>