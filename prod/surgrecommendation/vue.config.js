module.exports = {
  publicPath: './',
  devServer: {
    proxy: {
      '^/breastcancer': {
        target: 'http://localhost:8000',  // 后台接口域名
        changeOrigin: true,  //是否跨域
        pathRewrite: {
          '/breastcancer': ''
        }
      }
    }
  },
  css: {
    loaderOptions: {
      less: {
        // If you are using less-loader@5 please spread the lessOptions to options directly
        modifyVars: {
          'form-item-margin-bottom': '0px'
        },
        javascriptEnabled: true,
      },
    },
  },
};