import axios from 'axios'

export default function(json) {
	
	return axios({
		method: 'post',
		url: `${ process.env.VUE_APP_API_ENDPOINT }/runner`,
		data: json,
		headers: { 'Content-Type': 'application/json; charset=UTF-8\'', 'Access-Control-Allow-Origin': '*' },
	}).then(({ data }) => {
		return typeof data === 'object' ? data : JSON.parse(data)
	})
}
