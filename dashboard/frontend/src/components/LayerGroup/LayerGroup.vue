<template>
  <v-radio-group
    v-model="input"
    class="pa-4"
    @change="changed"
  >
    <v-radio
      v-for="layer in layers"
      :key="layer.id"
      :label="layer.name"
      :value="layer.id"
    />
  </v-radio-group>
</template>

<script>
  export default {
    data() {
      return { 
        layers: [
          {
            'name': 'Mean',
            'id': 'mean',
          }, 
          { 'name': 'STD',
            'id': 'std' } ],
        input: 'mean',
      }
    },
    methods: {
      changed(id) {
        console.log('@changed id', id)
      },
    },
  }

</script>
/* 
layers on props
on mounted take props read layers and set default layer to show
@change load new layer

 */