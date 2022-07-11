import Vue from 'vue'
import App from './App.vue'
import router from './router'
import axios from 'axios'


import Antd from 'ant-design-vue';
import 'ant-design-vue/dist/antd.less';
Vue.use(Antd);

Vue.prototype.$http = axios;

Vue.config.productionTip = false

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
 