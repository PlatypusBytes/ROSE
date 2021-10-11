<template>
  <!--  v-model="open" -->
  <v-expansion-panels class="mb-4">
    <v-expansion-panel>
      <v-expansion-panel-header
        color="primary"
        hide-actions
        class="font-weight-bold px-4"
      >
        {{ title }}
      </v-expansion-panel-header>
      <v-divider />
      <v-expansion-panel-content class="mx-n6">
        <slot name="slider" />
        <slot name="select" />
        <slot name="layers" />
      </v-expansion-panel-content>
    </v-expansion-panel>
  </v-expansion-panels>
</template>

<script>
 
  export default {
    props: {
      title: {
        type: String,
        required: false,
        default: '',
      },
      expand: {
        type: Number,
        default: null,
      },
    },
    data() {
      return {
        open: this.expand,
      }
    },

  }
</script>
