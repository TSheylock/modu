module.exports = {
  plugins: [
    require('postcss-flexbugs-fixes'),
    require('postcss-preset-env')({
      autoprefixer: {
        flexbox: 'no-2009'
      },
      stage: 3
    }),
    require('postcss-normalize'),
    require('postcss-custom-properties'),
    require('postcss-custom-media'),
    require('postcss-calc'),
    require('postcss-color-function'),
    require('postcss-discard-duplicates'),
    require('cssnano')({
      preset: ['default', {
        discardComments: {
          removeAll: true
        },
        normalizeWhitespace: false
      }]
    })
  ]
};
