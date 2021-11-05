<template>
  <div />
</template>

<script>
  import { mapActions } from 'vuex'

  export default {
    inject: [ 'getMap' ],
    props: {
      layer: {
        type: Object,
        default: null,
      },
    },
    data() {
      return {
        map: null,
      }
    },
    watch: {
      map: {
        handler(value) {
          if (value) {
            this.addEventsToMap()
          }
        },
      },
    },
    mounted() {
      const map = this.getMap()
      if (map) {
        this.map = map
      }
    },
    beforeDestroy() {
      this.removeEventsFromMap()
    },
    methods: {
      ...mapActions('charts', [ 'setChartData' ]),
      deferredMountedTo(map) {
        if (this.layer) {
          this.map = map
        }
      },
      onClick(e) {
        const { segmentId } = e.features[0].properties
        this.setChartData(segmentId)
      },
      onMouseEnter() {
        this.map.getCanvas().style.cursor = 'pointer'
      },
      onMouseLeave() {
        this.map.getCanvas().style.cursor = ''
      },
      addEventsToMap() {
        this.map.on('click', this.layer.id, this.onClick)
        this.map.on('mouseenter', this.layer.id, this.onMouseEnter)
        this.map.on('mouseleave', this.layer.id, this.onMouseLeave)
      },
      removeEventsFromMap() {
        this.map.off('click', this.layer.id, this.onClick)
        this.map.off('mouseenter', this.layer.id, this.onMouseEnter)
        this.map.off('mouseleave', this.layer.id, this.onMouseLeave)
      },
    },
  }
</script>
