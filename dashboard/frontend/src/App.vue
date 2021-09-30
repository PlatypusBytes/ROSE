<template>
  <v-app> 
    <v-stepper
      v-model="page"
    >
      <v-stepper-header>
        <v-stepper-step
          :complete="page >1"
          step="1"
        >
          Input data
        </v-stepper-step>
        <v-divider />
           
        <v-stepper-step
          :complete="page> 2"
          step="2"
        >
          Results
        </v-stepper-step>
      </v-stepper-header>
      
      <v-main>
        <v-stepper-items>
          <v-stepper-content step="1">
            <input-data-card @go-to-results="onShowResultsPage" />
          </v-stepper-content>
          <v-stepper-content step="2">
            <v-card
              class="pa-0 ml-0"
              height="92vh"
            >
              <mapbox-map :access-token="accessToken" />
            </v-card>
          </v-stepper-content>
        </v-stepper-items>
      </v-main>
    </v-stepper>
  </v-app>
</template>


<script>
  
  import { MapboxMap } from '@deltares/vue-components'
  import InputDataCard from '~/components/InputDataCard/InputDataCard'

  export default {
    components: {
      MapboxMap,
      InputDataCard,
    },
    data: () => ({
      accessToken: process.env.VUE_APP_MAPBOX_TOKEN,
      page: 1,
    }),
    methods: {
      onShowResultsPage(event) {
        this.page = event
      },
    },

  }
</script>

<style lang="scss" scoped>
/* 
  .v-stepper__header {
    height: 8vh;
  } */

  .v-stepper__content {
    padding: 0px;
  }

  .mapbox-map__title {
    position: absolute;
    z-index: 1;
    top: $spacing-default;
    left: $spacing-default;
    padding: $spacing-smaller $spacing-small;
    background-color: $color-white;
    user-select: none;
  }

  .mapbox-map__title .text-body-2 {
    margin: 0;
  }


  #map {
    z-index: 5;
  }

  .container {
    height: 800px;
  }
</style>
