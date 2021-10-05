<template>
  <v-app>
    <v-app-bar
      app
      color="white"
      height="72px"
      clipped-left
    >
      <v-toolbar-title>ROSE </v-toolbar-title>
      <v-spacer />
      <v-stepper
        v-model="page"
        flat
        width="80%"
        height="100%"
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
      </v-stepper>
    </v-app-bar>
    <v-navigation-drawer
      v-if="page===2"
      clipped
      app  
      permanent
      width="400"
    >
      <app-side-bar />
    </v-navigation-drawer> 
    <v-main app>
      <v-stepper
        v-model="page"
        flat
        class="pa-0"
      >
        <v-stepper-items>
          <v-stepper-content step="1">
            <input-data-card @go-to-results="onShowResultsPage" />
          </v-stepper-content>
          <v-stepper-content step="2" class="pa-0">
            <v-card
              class="pa-0"
              height="92vh"
            >
              <mapbox-map :access-token="accessToken" />
            </v-card>
          </v-stepper-content>
        </v-stepper-items>
      </v-stepper>
    </v-main>
  </v-app>
</template>


<script>

  import { MapboxMap } from '@deltares/vue-components'
  import InputDataCard from '~/components/InputDataCard/InputDataCard'
  import AppSideBar from '~/components/AppSideBar/AppSideBar'



  export default {
    components: {
      MapboxMap,
      InputDataCard,
      AppSideBar,
   
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
</style>
