<template>
  <v-expansion-panel>
    <v-expansion-panel-header
      class="font-weight-bold px-4"
      @transitionend="(event) => onTransitionEnd(event)"
    >
      {{ title }}
    </v-expansion-panel-header>
    <v-divider />
    <v-expansion-panel-content :ref="`layerType-${ id }`" class="mx-n6">
      <slot name="slider" />
      <slot name="trainTypes" />
      <slot name="valueTypes" />
    </v-expansion-panel-content>
  </v-expansion-panel>
</template>

<script>
  import { mapActions } from 'vuex'

  export default {
    props: {
      title: {
        type: String,
        required: true,
        default: '',
      },
      id: {
        type: String,
        required: true,
        defaut: '',
      },
    },
    data() { 
      return {
        panelRef: null,
        layerType: null, 
      }
    },
    watch: {
      panelRef() {
        console.log('this.panelRef i watcher', this.panelRef)
        const isActive = this.panelRef.isActive
        if (isActive) {
          this.setLayerType(this.id)
        }
      },
    },
    mounted() {
      this.panelRef = this.$refs[`layerType-${ this.id }`]
      console.log('this.panelRef i mounted', this.panelRef)
    },
    methods: { 
      ...mapActions('layers', [ 'setLayerType' ]),
      onTransitionEnd(event) {
        const isActive  = this.panelRef.isActive
        const { propertyName } = event
        if (propertyName === 'min-height') {
          return
        }
        if (isActive) {
          this.setLayerType(this.id)
        }
      },
    },
  }
</script>

