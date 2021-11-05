import axios from 'axios'

export default function(params) {
  
  return axios({
    method: 'get',
    url: `${ process.env.VUE_APP_API_ENDPOINT }/dynamic_stiffness`,
    params,
    proxy:{
      host:'localhost',
      port:5000,
    },
  }).then(({ data }) => {
		return typeof data === 'object' ? data : JSON.parse(data)
	})
}
