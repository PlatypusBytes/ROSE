import Vue from 'vue'
import './plugins/composition-api'
import App from './App.vue'
import store from './store'
import vuetify from './plugins/vuetify'

Vue.config.productionTip = false

import './components/AppCore/index.scss'

new Vue({
  store,
  vuetify,
  render: h => h(App),
}).$mount('#app')
