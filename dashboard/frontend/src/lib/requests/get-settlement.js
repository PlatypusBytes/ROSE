import axios from 'axios'

export default function(params) {
  const emptyParams = Object.values(params).some(param => param === null )
  if (emptyParams) {
    return 
  }
  return axios({
    method: 'get',
    url: `${ process.env.VUE_APP_API_ENDPOINT }/settlement`,
    params,
  }).then(({ data }) => {
    console.log('data get-settlement', JSON.stringify(data))
		return typeof data === 'object' ? data : JSON.parse(data)
	})
}