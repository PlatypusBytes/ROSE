<template>
  <v-card
    class="map-legend"
    flat
  >
    <v-card-title>
      {{ title }}
    </v-card-title>
    <v-card-text>
      <v-list dense>
        <v-list-item-group>
          <v-list-item v-for="(item, index) in items" :key="index">
            <v-list-item-icon>
              <v-sheet
                :color="item.color"
                height="24"
                rounded
                width="24"
              />
            </v-list-item-icon>
            <v-list-item-content>
              <span class="text-body-2">
                {{ item.label }}
              </span>
            </v-list-item-content>
          </v-list-item>
        </v-list-item-group>
      </v-list>
    </v-card-text>
  </v-card>
</template>

<script>

  export default {

    props: {
      items: {
        type: Array,
        default: () => [],
      },
      title: {
        type: String, 
        default: '',
      },
    },
  }
</script>

<style lang="scss">
  .map-legend {
    position: absolute;
    z-index: 2;
    right: $spacing-default;
    bottom: $spacing-large;
    pointer-events: none;
  }
</style>
