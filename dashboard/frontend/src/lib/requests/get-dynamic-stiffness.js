import axios from 'axios'

export default function(params) {
 
  const emptyParams = Object.values(params).some(param => param === null )
  if (emptyParams) {
    
    return 
  }
 
  return axios({
    method: 'get',
    url: `${ process.env.VUE_APP_API_ENDPOINT }/dynamic_stiffness`,
    params,
  }).then((response) => {
    console.log('response', response)
  })

}