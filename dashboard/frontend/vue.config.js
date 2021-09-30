const path = require('path')

module.exports = {
  css: {
    loaderOptions: {
      scss: {
        prependData: '@import \'~@/components/AppCore/variables.scss\';',
      },
    },
  },
  outputDir: path.resolve(__dirname, "../templates/"),
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.md$/,
          loader: 'raw-loader',
        },
      ],
    },
    resolve: {
      alias: {
        '~': path.resolve(__dirname, 'src/'),
      },
    },
  },
}

