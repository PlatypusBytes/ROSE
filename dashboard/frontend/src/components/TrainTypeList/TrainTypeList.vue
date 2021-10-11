/* 
Select train type commponent
 */
<template>
  <div class="pa-4">
    <div class="text-body-1 pb-3">
      Train type:
    </div>
    <v-select
      :items="trainTypes"
      :value="selectedType"
      outlined
      desnse
    />
  </div>
</template>

<script>
  export default {
    data() { 
      return { 
        trainTypes: [ 'typeA', 'typeB' ],
        selectedTyp: null,
      }
    },
  }
</script>