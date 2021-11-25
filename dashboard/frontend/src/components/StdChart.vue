<template>
  <v-chart
    class="line-chart"
    :init-options="initOptions"
    :option="options"
    autoresize
  />
</template>
<script>
  import { use } from 'echarts/core'
  import { CanvasRenderer } from 'echarts/renderers'
  import { LineChart } from 'echarts/charts'
  
  import {
    GridComponent,
    TitleComponent,
    TooltipComponent,
  } from 'echarts/components'
  import VChart from 'vue-echarts'
  
  use([
    CanvasRenderer,
    GridComponent,
    LineChart,
    TitleComponent,
    TooltipComponent,
  ])

  export default {
    name: 'StdChart',
    components: {
      VChart,
    },
    props: { 
      meanValues: {
        type: Array,
        required: true, 
        default: () => [],
      },
      minValues: {
        type: Array,
        required: true, 
        default: () => [],
      },
      maxValues: {
        type: Array,
        required: true, 
        default: () => [],
      },
      time: {
        type: Array,
        required: true, 
        default: () => [],
      },
    },
    data() { 
      return { 
        initOptions: { height: '400px' },
        baseOptions: {
          backgroundColor: 'rgb(243, 243, 243)',
          tooltip: {
            trigger: 'axis',
          },
        },
      }
    },
    computed: {
      options() {
        return {
          ...this.baseOptions,
          xAxis: this.xAxis,
          yAxis: this.yAxis,
          series: this.series,
        }
      },
      xAxis() {
        return {
          type: 'category',
          name: 'Time [d]',
          nameLocation: 'middle',
          nameGap: 30,
          data: this.time,
        }
      },
      yAxis() {
        return {
          splitLine: { show: false },
          type: 'value',
          name: 'Vertical displacement [mm]',
          nameLocation: 'middle',
          nameGap: 30,
          splitNumber: 8,
        }
      },
      series() { 
        return [ this.maxSeries, this.meanSeries, this.minSeries ]
        
      },
      maxSeries() { 
        return { 
          name: 'max',
          type: 'line',
          lineStyle: {
            type: 'dotted',
            opacity: 1,
          },
          areaStyle: { 

          },
          symbol: 'none',
          data: this.maxValues,
        }
      },
      meanSeries() { 
        return { 
          name: 'mean',
          type: 'line',
          itemStyle: { 
            color: '#333', 
          },
          showSymbol: false,
          data: this.meanValues,
        }
      },
      minSeries() { 
        return { 
          name: 'min',
          type: 'line',
          lineStyle: { 
            type: 'dotted',
            opacity: 1,
          },
          areaStyle: { 
            color: 'rgb(243, 243, 243)',
            opacity: 1,
          },
          symbol: 'none',
          data: this.minValues,
        }
      },
    },
  }
</script>