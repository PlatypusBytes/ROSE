import axios from 'axios'

export default function(params) {
  return axios({
    method: 'get',
    url: `${ process.env.VUE_APP_API_ENDPOINT }/settlement`,
    params,
  }).then(({ data }) => {
		return typeof data === 'object' ? data : JSON.parse(data)
	})
}