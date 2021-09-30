<template>
  <v-container fluid>
    <v-card
      class="mx-auto my-12"
      max-width="374"
      height="420"
    >
      <v-card-title>
        Input Data
      </v-card-title>
      <v-card-text>
        <v-file-input 
          label="Import SOS segments"
          hint="Compulsory"
          prepend-icon=""
        />
        <v-file-input
          label="Import Sensar"
          prepend-icon=""
          hint="Optional"
        />
        <v-file-input
          label="Import Rila"
          prepend-icon=""
          hint="Optional"
        />
        <v-file-input
          label="Import InfraMon"
          prepend-icon=""
          hint="Optional"
        />
      </v-card-text>
      <v-card-actions>
        <div class="flex-grow-1" />
        <v-btn color="primary" @click="$emit('go-to-results', 2)">
          RUN
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>
